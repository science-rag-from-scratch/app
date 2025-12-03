import re
from loguru import logger
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import granite_picture_description


class PDFProcessor:
    def __init__(self):
        self.converter = self.__load_pdf_converter()
        self.emb_tokenizer, self.emb_model, self.device = self.__load_embedder()

    def __load_pdf_converter(self) -> DocumentConverter:
        # Text extractor configs
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True

        pipeline_options.do_formula_enrichment = True
        pipeline_options.do_picture_description = True
        pipeline_options.picture_description_options = granite_picture_description
        pipeline_options.picture_description_options.prompt = ("Describe the picture.")
        pipeline_options.images_scale = 1
        pipeline_options.generate_picture_images = True

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

    def __load_embedder(self):
        emb_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-large")
        emb_model = AutoModel.from_pretrained("intfloat/multilingual-e5-large")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        emb_model = emb_model.to(device)
        emb_model.eval()

        return emb_tokenizer, emb_model, device

    @staticmethod
    def __average_pool(last_hidden_states, attention_mask):
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_states.size()).float()
        sum_embeddings = torch.sum(last_hidden_states * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1).clamp(min=1e-9)
        return sum_embeddings/sum_mask
    
    def embed_text(self, text: str, max_length: int = 512):
        inputs = self.emb_tokenizer(
            "passage: "+text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            ).to(self.device)
        with torch.no_grad():
            outputs = self.emb_model(**inputs)
            emb = self.__average_pool(outputs.last_hidden_state, inputs['attention_mask'])
            emb = F.normalize(emb, p=2, dim=1)
        return emb[0].cpu().numpy()

    @staticmethod
    def chunk_text(text, chunk_size: int = 1500, overlap: int = 200):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def pdf_to_text(self, pdf_path: str) -> str | None:
        try:
            doc = self.converter.convert(pdf_path)
        except Exception as e:
            logger.error(f"Error converting PDF to text: {e}")
            return None
        return doc.document.export_to_text()
