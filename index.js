import express from 'express';
import cors from 'cors';
import puppeteer from 'puppeteer';
import { writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import multer from 'multer';
import path from 'node:path';
import { PDFLoader } from "langchain/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "langchain/embeddings/openai";
import { FaissStore } from "langchain/vectorstores/faiss";
import { ChatOpenAI } from "langchain/chat_models/openai";
import { BufferMemory } from "langchain/memory";
import { ConversationalRetrievalQAChain } from "langchain/chains";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { JSONLoader, JSONLinesLoader } from "langchain/document_loaders/fs/json";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { CSVLoader } from "langchain/document_loaders/fs/csv";

const app = express();
app.use(cors());
app.use(express.json());

const __dirname = dirname(fileURLToPath(import.meta.url));

const storage = multer.diskStorage({
    destination: function (req, file, cb) {
        cb(null, 'documents');
    },
    filename: function (req, file, cb) {
        cb(null, file.fieldname + '-' + Date.now() + path.extname(file.originalname));
    }
});


const upload = multer({ storage: storage });

app.post('/scrape', upload.array('files'), async (req, res) => {
    const urls = JSON.parse(req.body.urls);
    const files = req.files;
    if (!urls && !files) {
        return res.status(400).send('No URLs or files provided');
    }

    try {
        if (urls) {
            for (const url of urls) {
                await scrapeURL(url);
            }
        }
        if (files) {
        }

        res.status(200).json({ message: 'Successfully processed' });
    } catch (error) {
        console.error('Processing error:', error);
        res.status(500).send('Error during processing');
    }
});

async function scrapeURL(url) {
    const browser = await puppeteer.launch();
    const page = await browser.newPage();
    await page.goto(url, { waitUntil: 'networkidle0' });
    const text = await page.evaluate(() => {
        let texts = [];
        const tags = ['p', 'span', 'a', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'article', 'section', 'blockquote', 'figcaption', 'td', 'caption', 'nav', 'label', 'summary', 'aside'];
        tags.forEach(tag => {
            const elements = Array.from(document.querySelectorAll(tag));
            elements.forEach(element => texts.push(element.innerText.trim()));
        });
        return texts.filter(text => text.length > 0).join('\n');
    });

    const urlObj = new URL(url);
    const fileName = urlObj.hostname + '.txt';
    const filePath = join(__dirname, 'documents', fileName);

    await writeFile(filePath, text);
    console.log(`Saved: ${fileName}`);

    await browser.close();
}

const getPDFs = async () => {
    try {
        const directoryLoader = new DirectoryLoader("./documents",
            {
                ".json": (path) => new JSONLoader(path),
                ".jsonl": (path) => new JSONLinesLoader(path),
                ".txt": (path) => new TextLoader(path),
                ".csv": (path) => new CSVLoader(path),
                ".pdf": (path) => new PDFLoader(path),
            }
        );

        const docs = await directoryLoader.load();

        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
            separators: ["\n"],
        });

        const splitDocs = await textSplitter.splitDocuments(docs);

        const embeddings = new OpenAIEmbeddings({
            temperature: 1,
            azureOpenAIApiKey: "",
            azureOpenAIApiVersion: "",
            azureOpenAIApiInstanceName: "",
            azureOpenAIApiDeploymentName: "",
        });
        console.log("Embedded successfully")
        const vectorStore = await FaissStore.fromDocuments(splitDocs, embeddings);

        const llm = new ChatOpenAI({
            temperature: 1,
            azureOpenAIApiKey: "",
            azureOpenAIApiVersion: "",
            azureOpenAIApiInstanceName: "",
            azureOpenAIApiDeploymentName: "",
            streaming: true
        });
        const memory = new BufferMemory({ memoryKey: "chat_history", returnMessages: true });

        const conversationChain = ConversationalRetrievalQAChain.fromLLM(llm, vectorStore.asRetriever(), { memory });

        console.log('Response generated...');

        return conversationChain;

    } catch (error) {
        console.error(error);
    }
}

app.post('/ask', async (req, res) => {
    const question = req.body.question;
    if (!question) {
        return res.status(400).json({ error: 'Question is required' });
    }

    try {
        const conversation = await getPDFs();
        const answer = await conversation?.call({ question });
        res.json({ answer: answer?.text });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});


const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
    console.log(`Server running on port ${PORT}`);
});
