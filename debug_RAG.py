import glob
import os
import fitz
import tabula
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration
)
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
import io
import chromadb
# import pdfplumber
# from qdrant_client import QdrantClient
# from qdrant_client.models import PointStruct, Distance, VectorParams

KNOWLEDGE_FOLDER = "E:\Workspace\Project\Healthcare-Agent\documents"

import base64
from tqdm import tqdm

# Create the directories
def create_directories(base_dir):
    directories = ["images", "text", "tables", "page_images"]
    for dir in directories:
        os.makedirs(os.path.join(base_dir, dir), exist_ok=True)

# Process tables
def process_tables(filepath, page, page_num, base_dir, items):
    try:
        tables = tabula.read_pdf(filepath, pages=page_num + 1, multiple_tables=True, output_format="json")
        if not tables:
            return
        for table_idx, table in enumerate(tables):
            # heading = "\n".join([" | ".join(map(str, table.keys()))])
            # table_content = "\n".join([" | ".join(map(str, row)) for row in table.values])
            # table_text = heading + "\n" + table_content
            # table_file_name = f"{base_dir}/tables/{os.path.basename(filepath)}_table_{page_num}_{table_idx}.txt"
            # with open(table_file_name, 'w') as f:
            #     f.write(table_text)
            # items.append({"page": page_num, "type": "table", "text": table_text, "path": table_file_name})

            bbox = (
                table['left'],
                table['top'],
                table['right'],
                table['bottom']
            )

            pix = page.get_pixmap(
                clip=fitz.Rect(bbox),
                dpi=300
            )

            pix.save(f"{base_dir}/tables/table_{table_idx}.png")
            items.append({"page": page_num, "type": "table",  "path": f"{base_dir}/tables/table_{table_idx}.png"})

    except Exception as e:
        print(f"Error extracting tables from page {page_num}: {str(e)}")

# Process text chunks
def process_text_chunks(filepath, text, text_splitter, page_num, base_dir, items):
    chunks = text_splitter.split_text(text)
    for i, chunk in enumerate(chunks):
        text_file_name = f"{base_dir}/text/{os.path.basename(filepath)}_text_{page_num}_{i}.txt"
        with open(text_file_name, 'w') as f:
            f.write(chunk)
        items.append({"page": page_num, "type": "text", "text": chunk, "path": text_file_name})

# Process images
# def process_images(doc, filepath, page, page_num, base_dir, items):
#     images = page.get_images(full=True)
#     for idx, image in enumerate(images):
#         xref = image[0]
#
#         pix = pymupdf.Pixmap(doc, xref)
#         image_name = f"{base_dir}/images/{os.path.basename(filepath)}_image_{page_num}_{idx}_{xref}.png"
#         pix.save(image_name)
#         with open(image_name, 'rb') as f:
#             encoded_image = base64.b64encode(f.read()).decode('utf8')
#         items.append({"page": page_num, "type": "image", "path": image_name, "image": encoded_image})

def process_images(doc, filepath, page, page_num, base_dir, items):
    VERTICAL_THRESHOLD = 5  # khoảng cách tối đa giữa 2 strip

    # Sort image by bounding box
    image_infos = []

    for image in page.get_images():

        xref = image[0]

        rects = page.get_image_rects(xref)

        if len(rects) == 0:
            continue

        rect = rects[0]

        image_infos.append({
            "xref": xref,
            "rect": rect,
            "image": image
        })

    unique_images = []
    seen = set()

    for info in image_infos:

        rect = info["rect"]

        key = (
            round(rect.x0, 1),
            round(rect.y0, 1),
            round(rect.x1, 1),
            round(rect.y1, 1)
        )

        if key in seen:
            continue

        seen.add(key)
        unique_images.append(info)

    image_infos = unique_images

    # sort từ trên xuống dưới
    image_infos.sort(
        key=lambda x: (
            round(x["rect"].y0, 2),
            round(x["rect"].x0, 2)
        )
    )

    idx = 0

    while idx < len(image_infos):

        current = image_infos[idx]

        current_xref = current["xref"]
        current_rect = current["rect"]

        # khởi tạo group
        group = [current_xref]
        group_rects = [current_rect]

        next_idx = idx + 1

        while next_idx < len(image_infos):

            next_item = image_infos[next_idx]

            next_xref = next_item["xref"]
            next_rect = next_item["rect"]

            prev_rect = group_rects[-1]

            ################################################
            # điều kiện merge
            ################################################

            width_similar = (
                    abs(prev_rect.width - next_rect.width) < 20
            )

            same_left = (
                    abs(prev_rect.x0 - next_rect.x0) < 5
            )

            same_right = (
                    abs(prev_rect.x1 - next_rect.x1) < 5
            )

            vertical_gap = (
                    next_rect.y0 - prev_rect.y1
            )

            close_vertical = (
                    abs(vertical_gap) <= VERTICAL_THRESHOLD
            )

            ################################################
            # merge
            ################################################

            if (
                    width_similar
                    and same_left
                    and same_right
                    and close_vertical
            ):

                group.append(next_xref)
                group_rects.append(next_rect)
                next_idx += 1

            else:
                break



        #################################################################
        # merge image
        #################################################################

        pil_images = []

        for xref in group:
            pix = fitz.Pixmap(doc, xref)

            img = Image.open(
                io.BytesIO(pix.tobytes("png"))
            )

            pil_images.append(img)


        if len(pil_images) == 1:

            merged = pil_images[0]

        else:

            total_height = sum(
                img.height for img in pil_images
            )

            max_width = max(
                img.width for img in pil_images
            )

            merged = Image.new(
                "RGB",
                (max_width, total_height)
            )

            y = 0

            for img in pil_images:
                merged.paste(img, (0, y))
                y += img.height

        #################################################################
        # save merged image
        #################################################################

        image_name = (
            f"{base_dir}/images/"
            f"{os.path.basename(filepath)}"
            f"_image_{page_num}_{idx}.png"
        )

        merged.save(image_name)

        with open(image_name, "rb") as f:
            encoded = base64.b64encode(
                f.read()
            ).decode("utf8")

        items.append({
            "page": page_num,
            "type": "image",
            "path": image_name,
            "image": encoded,
            "merged_xrefs": group
        })

        idx = next_idx

# Process page images
def process_page_images(page, page_num, base_dir, items):
    pix = page.get_pixmap()
    page_path = os.path.join(base_dir, f"page_images/page_{page_num:03d}.png")
    pix.save(page_path)
    with open(page_path, 'rb') as f:
        page_image = base64.b64encode(f.read()).decode('utf8')
    items.append({"page": page_num, "type": "page", "path": page_path, "image": page_image})

def describe_table(path):
    image = Image.open(path)

    messages = [{
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": image
            },
            {
                "type": "text",
                "text": TABLE_PROMPT
            }
        ]
    }]

    text = processor \
        .apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=512
    )

    generated_ids = outputs[
        0,
        inputs.input_ids.shape[1]:
    ]

    response = processor.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return response


def describe_figure(path):

    image = Image.open(path)

    messages = [{
        "role":"user",
        "content":[
            {
                "type":"image",
                "image":image
            },
            {
                "type":"text",
                "text":PROMPT
            }
        ]
    }]

    text = processor\
        .apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    inputs = processor(
        text=[text],
        images=[image],
        return_tensors="pt"
    ).to(model.device)

    # outputs = model.generate(
    #     **inputs,
    #     max_new_tokens=512
    # )
    #
    # return processor.decode(
    #     outputs[0],
    #     skip_special_tokens=True
    # )

    outputs = model.generate(
        **inputs,
        max_new_tokens=512
    )

    generated_ids = outputs[
        0,
        inputs.input_ids.shape[1]:
    ]

    response = processor.decode(
        generated_ids,
        skip_special_tokens=True
    )

    return response

def summarize_table(text, llm):

    prompt = f"""

    {TABLE_PROMPT}

    Table:

    {text}

    """

    response = llm.invoke(
        prompt
    )

    return response

for filepath in glob.glob(os.path.join(KNOWLEDGE_FOLDER, "*.pdf"))[:1]:
    filepath = "E:/Workspace/Project/Healthcare-Agent/documents/bioengineering-12-00286.pdf"
    doc = pymupdf.open(filepath)
    num_pages = len(doc)
    base_dir = "data"

    # Creating the directories
    create_directories(base_dir)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=200, length_function=len)
    items = []

    # Process each page of the PDF
    for page_num in tqdm(range(num_pages), desc="Processing PDF pages"):
        page_num = 5
        page = doc[page_num]

        text = page.get_text()
        # process_tables(filepath, page, page_num, base_dir, items)
        process_text_chunks(filepath, text, text_splitter, page_num, base_dir, items)
        # process_images(doc, filepath, page, page_num, base_dir, items)
        # process_page_images(page, page_num, base_dir, items)
        break

    # print(items)
    # Looking at the first text item
    # print([i for i in items if i['type'] == 'text'][0])

    # Looking at the first table item
    # print([i for i in items if i['type'] == 'table'][0])

    # Looking at the first image item
    # print([i for i in items if i['type'] == 'image'][0])



    model = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", torch_dtype="auto", device_map="auto")

    processor = AutoProcessor\
                    .from_pretrained(
                        "Qwen/Qwen2-VL-2B-Instruct"
                    )


    PROMPT = """
    Analyze this scientific figure for retrieval purposes.

    Provide:

    1. Figure Type
       (chart, flowchart, architecture, pipeline,
        medical image, waveform, heatmap,
        confusion matrix, ROC curve, etc.)

    2. Main Content
       Describe what the figure shows.

    3. Important Elements
       List:
       - labels
       - legends
       - axes
       - modules
       - blocks
       - variables
       - annotations

    4. Key Findings
       Summarize the main scientific insights.

    5. Retrieval Summary
       Write a concise paragraph (50-100 words)
       suitable for semantic search.

    6. Keywords
       List 10-20 technical keywords.
    """

    # for doc in items:
    #
    #     if doc["type"] == "image":
    #
    #         doc["description"] = \
    #             describe_figure(
    #                 doc["path"]
    #             )
    #
    #         print(doc["description"])
    #         print("=" * 50)

    TABLE_PROMPT = """
    Analyze this scientific table.

    Describe:

    - variables
    - statistics
    - trends
    - imbalance
    - important findings
    """

#     llm = Ollama(
#     model="qwen3:8b",
#     request_timeout=600
# )
#
#     for doc in items:
#
#         if doc["type"] == "table":
#             doc["summary"] = summarize_table(doc["text"], llm)
#             print("doc['summary']")
#             print(doc["summary"])
#             print("=" * 50)

    for doc in items:

        if doc["type"] == "table":

            doc["summary"] = \
                describe_table(
                    doc["path"]
                )

            print(doc["summary"])
            print("=" * 50)

    # PHASE 4 — Build canonical scientific chunk
    for doc in items:

        if doc["type"] == "text":

            doc["chunk"] = f"""
            TYPE: TEXT

            {doc['content']}
            """

        elif doc["type"] == "image":

            doc["chunk"] = f"""
            TYPE: FIGURE

            {doc['description']}
            """

        else:

            doc["chunk"] = f"""
            TYPE: TABLE

            {doc['summary']}
            """

            print("doc['chunk']")
            print(doc["chunk"])

    # PHASE 5 — Embedding
    embedder =  SentenceTransformer("BAAI/bge-small-en-v1.5")

    for doc in items:
        doc["embedding"] = embedder.encode(doc["chunk"], normalize_embeddings=True).tolist()


    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="scientific_papers",
        metadata={
            "hnsw:space": "cosine"
        }
    )

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for i, doc in enumerate(items):
        ids.append(str(i))

        embeddings.append(
            doc["embedding"]
        )

        documents.append(
            doc["chunk"]
        )

        metadatas.append({
            "doc_id": doc["path"],
            "type": doc["type"]
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas
    )

    # query = """
    # Which frequency bands are used in filtering EEG signals?
    # """

    query = """
        Which features are extracted in Time domain?
        """

    query_emb = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=[query_emb],
        n_results=3
    )

    for i in range(
            len(results["ids"][0])):
        print(
            "ID:",
            results["ids"][0][i]
        )

        print(
            "Score:",
            results["distances"][0][i]
        )

        print(
            "Type:",
            results["metadatas"][0][i]["type"]
        )

        print(
            results["documents"][0][i][:500]
        )

        print("=" * 50)

    print(results)
    """
    client = QdrantClient(
        "localhost",
        port=6333
    )

    client.create_collection(

        collection_name=
        "scientific_papers",

        vectors_config=
        VectorParams(

            size=1024,

            distance=
            Distance.COSINE
        )
    )

    points = []

    for i, doc in enumerate(items):
        points.append(

            PointStruct(

                id=i,

                vector=
                doc["embedding"],

                payload={

                    "doc_id":
                        doc["id"],

                    "type":
                        doc["type"],

                    "text":
                        doc["chunk"]
                }
            )
        )

    client.upsert(

        collection_name=
        "scientific_papers",

        points=points
    )

    query = "Which frequency bands are used in filtering EEG signals?"

    query_emb = embedder.encode(query)

    response = client.query_points(
        collection_name="my_collection",
        query=query_emb,  # Your query vector
        limit=3  # Number of top results to return
    )

    print(response)

    # Accessing results
    for point in response.points:
        print(f"ID: {point.id}, Score: {point.score}")
    """