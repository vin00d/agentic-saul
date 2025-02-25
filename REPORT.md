# 1. The problem, solution, audience
- A mid-sized legal firm with 100+ lawyers
- The firm has a large amount of legal documents, including contracts, case law, and legal opinions.
- Current processes are very manual, time-consuming, and error-prone with lawyers and paralegals spending a lot of time on repetitive tasks.
- High staff turnover and difficulty in training new employees plus high labor costs is adding to the firm's operational challenges.
- The firm is looking for ways to improve efficiency and reduce costs.
- The firm is exploring the use of AI to improve its legal research and document review processes.

## 1.1. Solution - Better Call Agentic-Saul
- A multi agent system finetuned on the firm's legal documents to provide a more efficient and accurate way to conduct legal research and document review.
- Available as a chatbot interface that can be accessed by lawyers, paralegals and even interns and potentially clients.
- The chatbot can answer basic to complex legal questions & perform deep research on legal topics.


### Agents 
- 📄 Legal Glossary - legal terms and definitions -- RAG over legal dictionaries       
- 📚 Wikipedia - basic information -- summary of `broad trends` like "Roe vs Wade" or "Presidential Election Results"
- 💬 Reddit discussions - current chatter in social media -- generally triggered by asking `What are people saying about this?`
- 📖 Google Scholar Case Law - judicial opinions from numerous federal and state courts -- specific case search and summarization

### Tech Stack
- LLM - OpenAI GPT-4o-mini
- Embeddings - OpenAI text-embedding-3-small (finetuned snowflake-arctic-embed-l not deployed)
- Vector DB - Qdrant
- Orchestration / RAG - LangChain
- Evaluation - RAGAS
- Deployment - Chainlit
- Serving - HuggingFace Spaces

### Short Demo
- [Loom Video](https://www.loom.com/share/4f04e72f9bd24433b1d129c18c5f327b?sid=e1b95f0a-1917-4648-8846-03627ac18b37)

# 2. Description of data
## 2.1. RAG
- Glossary of common legal terms - [NY Courts](http://www.nycourts.gov/lawlibraries/glossary.shtml)
- Glossary of legal terms - US Courts
- Black's Law Dictionary - comprehensive entries on legal terms - not used in this iteration
- Chunking strategy = `chunk_size = 750, chunk_overlap  = 50` 
    - since this is a smallish dataset mostly of definitions of terms, not long passages
    - later when using the bigger law dictionary this will need to be reconsidered for larger inputs to improve efficiency

## 2.2. Agents
- Wikipedia & Reddit - Langchain's integrations
- Google Scholar Case Law - SerpApi custom built tool

# 3. An End-to-End Agentic RAG prototype
- App deployed in Hugging Face Spaces - https://huggingface.co/spaces/vin00d/agentic-saul

# 4. Creating a Golden Test Data Set
**Eval Metrics for Base Model**
- Faithfulness and Answer Relevancy stand out as particularly problematic
- Legal terms are indeed unique with their use of Latin in legal contexts which are not in regular use and therefore unseen by AI models in their training data.

| Metric                      | Value   |
|-----------------------------|---------|
| Context Recall              | 0.2583  |
| Faithfulness                | <span style="color:red">0.3967  </span> |
| Factual Correctness         | 0.6225  |
| Answer Relevancy            | <span style="color:red">0.3189  </span> |
| Context Entity Recall       | 0.2639  |
| Noise Sensitivity Relevant  | 0.0354  |

# 5. Fine-Tuning Open-Source Embeddings
- Finetuned model - https://huggingface.co/vin00d/snowflake-arctic-legal-ft-1

# 6. Assessing Performance
**Eval Metrics for Fine-Tuned Model**
- The fine-tuned model shows a significant improvement across all metrics, with the most notable improvements in Context Recall, Faithfulness and Answer Relevancy.

| Metric                      | Value   |
|-----------------------------|---------|
| Context Recall              | <span style="color:green">0.7667</span> |
| Faithfulness                | <span style="color:green">0.7221</span> |
| Factual Correctness         | 0.6925  |
| Answer Relevancy            | <span style="color:green">0.8779</span> |
| Context Entity Recall       | 0.3065  |
| Noise Sensitivity Relevant  | 0.1734  |

## 6.1. Next Steps
- More RAG data - Black Law's Dictionary, Law Review Articles, Case Law
- More Agents - Reuters APIs
- Production Grade Deployment & Monitoring

# Managing User Expectations
- This chatbot while only a prototype is a significant improvement over the current manual processes.
- It is not a replacement for human lawyers, but rather a tool to assist them in their work.
- The chatbot is not perfect and will make mistakes, but it is constantly learning and improving.
- The chatbot is not a substitute for legal advice & human judgement and should not be used as such.