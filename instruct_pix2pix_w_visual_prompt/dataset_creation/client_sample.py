from msal import PublicClientApplication, SerializableTokenCache
import json
import os
import atexit
import requests


class LLMClient:

    _ENDPOINT = 'https://httpqas26-frontend-qasazap-prod-dsm02p.qas.binginternal.com/completions'
    _SCOPES = ['api://68df66a4-cad9-4bfd-872b-c6ddde00d6b2/access']

    def __init__(self):
        self._cache = SerializableTokenCache()
        atexit.register(lambda: 
            open('.llmapi.bin', 'w').write(self._cache.serialize())
            if self._cache.has_state_changed else None)

        self._app = PublicClientApplication('68df66a4-cad9-4bfd-872b-c6ddde00d6b2', authority='https://login.microsoftonline.com/72f988bf-86f1-41af-91ab-2d7cd011db47', token_cache=self._cache)
        if os.path.exists('.llmapi.bin'):
            self._cache.deserialize(open('.llmapi.bin', 'r').read())

    def send_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers)
        return response.json()

    def send_stream_request(self, model_name, request):
        # get the token
        token = self._get_token()

        # populate the headers
        headers = {
            'Content-Type':'application/json', 
            'Authorization': 'Bearer ' + token, 
            'X-ModelType': model_name }

        body = str.encode(json.dumps(request))
        response = requests.post(LLMClient._ENDPOINT, data=body, headers=headers, stream=True)
        for line in response.iter_lines():
            text = line.decode('utf-8')
            if text.startswith('data: '):
                text = text[6:]
                if text == '[DONE]':
                    break
                else:
                    yield json.loads(text)       

    def _get_token(self):
        accounts = self._app.get_accounts()
        result = None

        if accounts:
            # Assuming the end user chose this one
            chosen = accounts[0]

            # Now let's try to find a token in cache for this account
            result = self._app.acquire_token_silent(LLMClient._SCOPES, account=chosen)

    
        if not result:
            # So no suitable token exists in cache. Let's get a new one from AAD.
            flow = self._app.initiate_device_flow(scopes=LLMClient._SCOPES)

            if "user_code" not in flow:
                raise ValueError(
                    "Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

            print(flow["message"])

            result = self._app.acquire_token_by_device_flow(flow)

        return result["access_token"]

llm_client = LLMClient()

# request_data = {
#         "prompt":"Let us play a game. Based on some examples, fill in the next sentence. input: Sicily, Impression by micorl, edit: make it become photorealistic. input: First Snow Surrey Hills (acrylic ink)_, edit:",
#         "max_tokens":500,
#         "temperature":0.6,
#         "top_p":1,
#         "n":5,
#         "stream":False,
#         "logprobs":None,
#         "stop":"\n"
# }

# get the model response
# available models are:
# text-davinci-001 (GPT-3)
# text-davinci-002 (GPT-3.5)
# text-davinci-003 (GPT-3.51)
# text-chat-davinci-002 (ChatGPT)
# response = llm_client.send_request('text-davinci-003', request_data)
# response = llm_client.send_request('text-chat-davinci-002', request_data)
#
# print(response['choices'][0]['text'])

# prompt = "Instruction: find the entity of the sentence. \nInput: swap with Oprah Winfrey"
# prompt = "Instruction: Add some descriptive modifier to this word. Describe action, color, clothes or shape. \nInput: Oprah Winfrey"
# stream_request_data = {
#         "prompt": "Instruction: Given an input question, respond with syntactically correct c++. Be creative but the c++ must be correct. \nInput: Create a function in c++ to remove duplicate strings in a std::vector<std::string>\n",
#         # "prompt": prompt,
#         "max_tokens":500,
#         "temperature":0.6,
#         "top_p":1,
#         "n":1,
#         "stream":True,
#         "logprobs":None,
#         "stop":"\r\n"
# }
#
# for response in llm_client.send_stream_request('text-davinci-003', stream_request_data):
#     print(response['choices'][0]['text'], end='')

