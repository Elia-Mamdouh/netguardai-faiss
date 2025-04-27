import os
import json
import re
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
app = Flask(__name__)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

# Load dataset
with open("final_dataset.json", "r", encoding="utf-8") as f:
    dataset = json.load(f)

vendor_mapping = {
    "cisco": "Cisco",
    "juniper": "Juniper",
    "palo alto": "Palo Alto",
    "fortinet": "Fortinet",
    "f5": "F5"
}

vendor_setup_stores = {}
vendor_security_stores = {}
user_contexts = {}
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
all_commands = []

def build_vendor_docs():
    setup_docs, security_docs = {}, {}

    def collect_features(vendor, subtree, device_path):
        if not isinstance(subtree, dict): return
        if "setup" in subtree:
            for cmd in subtree["setup"]:
                all_commands.append({"vendor": vendor, "type": "setup", "device": device_path, **cmd})
                text = f"Vendor: {vendor}\nDevice: {device_path}\nType: Setup\nName: {cmd.get('name')}\nDescription: {cmd.get('description')}\nCommand: {cmd.get('command')}\n"
                setup_docs[vendor].append(Document(page_content=text))
        if "security" in subtree:
            for level, commands in subtree["security"].items():
                for cmd in commands:
                    all_commands.append({"vendor": vendor, "type": "security", "level": level, "device": device_path, **cmd})
                    text = f"Vendor: {vendor}\nDevice: {device_path}\nSecurity Level: {level}\nName: {cmd.get('name')}\nDescription: {cmd.get('description')}\nCommand: {cmd.get('command')}\nRecommendations: {cmd.get('recommendations', 'None')}\n"
                    security_docs[vendor].append(Document(page_content=text))
        for key, val in subtree.items():
            if key.lower() not in {v.lower() for v in vendor_mapping.values()}:
                collect_features(vendor, val, device_path + "/" + key)

    for vendor, devices in dataset["vendors"].items():
        setup_docs[vendor], security_docs[vendor] = [], []
        collect_features(vendor, devices, vendor)

    return setup_docs, security_docs

def build_faiss_indexes():
    setup_docs, security_docs = build_vendor_docs()
    for vendor in setup_docs:
        vendor_setup_stores[vendor] = FAISS.from_documents(setup_docs[vendor], embeddings) if setup_docs[vendor] else None
    for vendor in security_docs:
        vendor_security_stores[vendor] = FAISS.from_documents(security_docs[vendor], embeddings) if security_docs[vendor] else None

build_faiss_indexes()

@app.route("/query", methods=["POST"])
def query_faiss():
    data = request.get_json()
    user_query = data.get("query", "").lower().strip()
    user_id = data.get("user_id", "default_user")
    preview_mode = "preview" in user_query or "impact" in user_query

    irrelevant_keywords = [
    "joke", "weather", "powerpoint", "excel", "word", "story", "open", "time", "date", 
    "calendar", "meme", "music", "restaurant", "movie", "tell me", "funny", "game"
    ]

    if any(word in user_query for word in irrelevant_keywords):
        return jsonify({"results": ["⚠️ Sorry, I can only assist you with network setup and security topics related to setting up or securing any of the devices across the 5 platforms"]})

    if any(greet in user_query for greet in ["hello", "hi", "hey"]):
        return jsonify({"results": ["Hello! How can I assist you with network setup or security today?"]})

    if preview_mode:
        return jsonify({"results": [get_preview(user_query)]})

    if any(k in user_query for k in ["set", "enable", "disable", "configure"]):
        return jsonify({"results": [handle_scenario(user_query)]})

    if "features" in user_query:
        vendors_found = detect_vendors(user_query, user_id)
        level_filter = "advanced" if "advanced" in user_query else ("basic" if "basic" in user_query else None)
        if "security" in user_query:
            return jsonify({"results": [generate_feature_list(vendors_found, "security", level_filter)]})
        elif "setup" in user_query:
            return jsonify({"results": [generate_feature_list(vendors_found, "setup")]})

    mentioned_vendors = detect_vendors(user_query, user_id)
    is_setup = ("setup" in user_query) or ("configure" in user_query)
    all_results = []
    for vendor in mentioned_vendors:
        store = vendor_setup_stores.get(vendor) if is_setup else vendor_security_stores.get(vendor)
        if store:
            docs = store.similarity_search(user_query, k=5)
            all_results.extend([d.page_content for d in docs])
    if not all_results:
        return jsonify({"results": ["Sorry, I couldn't find anything useful for that request."]})
    return jsonify({"results": all_results})

def detect_vendors(user_query, user_id):
    mentioned = [vendor_mapping[k] for k in vendor_mapping if k in user_query]
    if not mentioned and user_id in user_contexts:
        mentioned = [user_contexts[user_id]]
    if not mentioned:
        mentioned = list(vendor_mapping.values())
    user_contexts[user_id] = mentioned[0]
    return mentioned

def generate_feature_list(vendors_found, feature_type, level_filter=None):
    response_text = ""
    for vendor in vendors_found:
        if vendor not in dataset["vendors"]:
            continue
        response_text += f"\n✨ {feature_type.capitalize()} Features for {vendor}:\n"
        def extract_features(subtree, device_path):
            nonlocal response_text
            if not isinstance(subtree, dict): return
            if feature_type in subtree:
                for level, commands in subtree[feature_type].items() if feature_type == "security" else [("", subtree[feature_type])]:
                    if level_filter and level.lower() != level_filter: continue
                    label = device_path.split("/")[-1]
                    response_text += f"\n🔐 {label} {'- ' + level.capitalize() + ' Level' if level else ''}:\n"
                    for command in commands:
                        response_text += f"- {command['name']}: {command['description']}\n"
            else:
                for k, v in subtree.items():
                    if k.lower() not in [v.lower() for v in dataset["vendors"].keys()]:
                        extract_features(v, device_path + "/" + k)
        extract_features(dataset["vendors"][vendor], vendor)
    response_text += "\nWould you like to configure another feature? All commands can be exported if needed."
    return response_text

def handle_scenario(query):
    matched = []
    for cmd in all_commands:
        terms = [cmd['name'], cmd.get('description', '')]
        if any(term.lower() in query for term in terms):
            filled = cmd["command"]
            for key in re.findall(r"\[(.*?)\]", filled):
                guess = re.search(key.replace("_", " "), query)
                filled = filled.replace(f"[{key}]", guess.group(0) if guess else f"[{key}]")
            matched.append(f"{cmd['name']}\n{filled}")
    if not matched:
        return "I understood that you're trying to configure something, but I need a bit more detail. Try using keywords like SSH, VLAN, hostname, or IP."
    return "Here are the configuration commands based on your scenario:\n\n" + "\n\n".join(matched)

def get_preview(query):
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [
                    {"role": "system", "content": "You are a network security expert. Provide benefits and recommendations for the following network security feature:"},
                    {"role": "user", "content": query}
                ],
                "temperature": 0.7
            },
            timeout=30
        )
        res_json = response.json()
        content = res_json["choices"][0]["message"]["content"]
        return f"🔍 **Preview:**\n{content.strip()}"
    except Exception as e:
        return f"⚠️ Failed to generate preview: {str(e)}"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)
