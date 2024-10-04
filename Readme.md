# AI Candidate Evaluator

This repository provides a powerful AI-based tool to assist recruiters in evaluating job candidates. It uses advanced AI techniques to analyze candidate resumes and match them against job descriptions, generating insightful answers to recruiters' questions. The code implements the Retrieval-Augmented Generation (RAG) approach to extract relevant candidate information and match it against the job's requirements.

## Features

- **Resume and Job Description Analysis**: Upload candidate resumes and job descriptions to extract key information.
- **Intelligent Question Answering**: The AI can answer recruiter queries about candidates and how they align with the job description.
- **Retrieval-Augmented Generation (RAG)**: Documents are split into chunks, embedded using OpenAI embeddings, and stored in a FAISS vector store for efficient retrieval.
- **Structured Output**: The system provides a structured analysis of candidates based on their resumes and job descriptions.
