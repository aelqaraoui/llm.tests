import * as dotenv from "dotenv";
import fs from "fs";
import { Configuration, OpenAIApi } from "openai";
dotenv.config();

const configuration = new Configuration({
  apiKey: process.env.OPENAI_API_KEY,
});
const openai = new OpenAIApi(configuration);

const get_completion = async (prompt: string) => {
  const completion = await openai.createChatCompletion({
    model: "gpt-3.5-turbo",
    messages: [{ role: "user", content: prompt }],
    temperature: 0,
  });

  return completion.data.choices[0].message?.content;
};

const get_embeddings = async (prompt: string) => {
  const response = await openai.createEmbedding({
    model: "text-embedding-ada-002",
    input: prompt,
  });

  return response.data.data[0].embedding;
};

function dot(a: number[], b: number[]): number {
  let dotProduct = 0;
  for (let i = 0; i < a.length; i++) {
    dotProduct += a[i] * b[i];
  }
  return dotProduct;
}

function norm(vector: number[]): number {
  let sumOfSquares = 0;
  for (let i = 0; i < vector.length; i++) {
    sumOfSquares += Math.pow(vector[i], 2);
  }
  return Math.sqrt(sumOfSquares);
}

function cosine_similarity(a: number[], b: number[]): number {
  return dot(a, b) / (norm(a) * norm(b));
}

function average(array: number[]): number {
  let sum = 0;
  for (let i = 0; i < array.length; i++) {
    sum += array[i];
  }
  return sum / array.length;
}

const data = fs.readFileSync("demo_data.txt", "utf-8");

const pattern = /Q: (.*?)\nA: (.*?)\n/gs;

let result: RegExpMatchArray | null;

const dataset: Array<{ prompt: string; response: string }> = [];

while ((result = pattern.exec(data)) !== null) {
  const [_, question, answer] = result;
  dataset.push({
    prompt: question,
    response: answer,
  });
}

console.log(dataset);

let tools_desc =
  'deposit_crypto : \n        Use this tool when the user wants to deposit crypto into the exchange.\n        Input format : ```{ \n            "crypto": the cryptocurrency you want to deposit, should be a string\n        }```\n        \n\nwithdrawl_crypto : \n        Use this tool when the user wants to withdrawl crypto from the exchange.\n        Input format : ```{ \n            "crypto": the cryptocurrency you want to withdrawl, should be a string, \n            "amount": the amount of cryptocurrency to withdrawl, should be an int or float, \n            "address": the wallet address you want to address to, should be a string\n        }```\n        \n\ntrade_crypto : \n        Use this tool when the user wants to trade a cryptocurrency pair.\n        Input format : ```{ \n            "pair": the cryptocurrency pair you want to trade, should be a string, \n            "action": the action to take, either BUY or SELL, should be a string, \n            "amount": the amount of crypto to trade, should be an int or float\n        }```\n        \n\ntalk_to_user : \n        Use this tool when you want info from the user or wan to talk to the user.\n        Input format : ```{\n            "message": the message or question you want the user to see, should be a string\n        }```\n        ';
let tools_names =
  "deposit_crypto, withdrawl_crypto, trade_crypto, talk_to_user";

let test_prompt = async () => {
  let similarities = [];

  await Promise.all(
    dataset.map(async (data) => {
      let user_input = data["prompt"];
      let history = `User Input: \`\`\`${user_input}\`\`\``;

      let prompt = `You are a helpful crypto exchange assistant.
You will be chating with a crypto user and need to help him answer crypto questions, deposit, withdrawl and trade crypto. 

If you don't have all the info to use a tool make sure to ask the user for the additional info.

Here are the tools available to you: 
${tools_desc}

You will be provided with the history of the conversation between you and the user in the following format:

User Input: the user specifies the task or gives feedback on the previous actions taken
Tool: the action to take
Tool Input: the input to the action
Observation: the result of the action
... (this User Input/Tool/Tool Input/Message/Observation can repeat N times)

Respond using the following format (JSON):

{ "tool": the action to take, should be one of [${tools_names}], "tool_input": the input to the action, "message": "the message the user sees" }

The response needs to be able to be JSON parsable.
You only need to respond with the next JSON.

Begin!

History: 
${history}

Response:    
`;

      let completion = await get_completion(prompt);

      if (completion) {
        console.log(completion);

        let similarity = cosine_similarity(
          await get_embeddings(completion),
          await get_embeddings(data["response"])
        );

        console.log(similarity);

        similarities.push(similarity);
      }

      console.log(
        "The average similarity score for example 1 on the testset : ",
        average(similarities)
      );
    })
  );
};

test_prompt();
