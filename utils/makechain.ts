import { OpenAI } from 'langchain/llms/openai';
import { Chroma } from 'langchain/vectorstores/chroma';
import { ConversationalRetrievalQAChain } from 'langchain/chains';

// Enum for model names
enum ModelName {
  GPT3 = 'gpt-3.5-turbo-0613',
  GPT4 = 'gpt-4'
}

const CONDENSE_PROMPT = `Given the conversation history and a follow-up question, rephrase the follow-up question to stand on its own while still referencing the necessary context from the conversation. 

Chat History:
{chat_history}
Follow Up Input: {question}
Rephrased standalone question:`;

const QA_PROMPT = `You are a helpful AI assistant. Given the context below, answer the final question. If the answer isn't clear from the context, or if you don't know the answer, simply state that you don't know. 

{context}

Question: {question}
Your detailed answer in markdown format:`;

export const makeChain = (
  vectorstore: Chroma,
  temperature: number = .3, 
  modelName: ModelName = ModelName.GPT4
) => {
  let model;

  try {
    model = new OpenAI({
      temperature,
      modelName
    });
  } catch (error) {
    console.warn("Failed to use GPT-4, falling back to GPT-3: ", error);

    model = new OpenAI({
      temperature,
      modelName: ModelName.GPT3
    });
  }

  const chain = ConversationalRetrievalQAChain.fromLLM(
    model,
    vectorstore.asRetriever(),
    {
      qaTemplate: QA_PROMPT,
      questionGeneratorTemplate: CONDENSE_PROMPT,
      returnSourceDocuments: true,
    },
  );
  
  return chain;
};
