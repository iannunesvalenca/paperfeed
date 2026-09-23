"""Offline fixtures: patch PaperFeed's collectors with generated papers (no network)."""
import random
from datetime import date, timedelta

import app

TITLES = [
    ("Long-read nanopore metagenomics resolves carbapenemase plasmids in hospital wastewater", "metagenomic"),
    ("Unbiased metagenomic sequencing for viral diagnosis in cerebrospinal fluid", "virus"),
    ("A deep learning model predicts antimicrobial resistance from Illumina whole genome sequencing", "machine learning"),
    ("Gut microbiome dynamics during an influenza outbreak in long-term care", "microbiome"),
    ("Emerging infectious disease surveillance with shotgun sequencing of airport wastewater", "outbreak"),
    ("Transformer language models for protein function annotation", "transformer"),
    ("Rare disease diagnosis with exome and whole genome sequencing in 4,000 families", "exome"),
    ("A reproducible bioinformatics pipeline for viral genome assembly", "pipeline"),
    ("Sequencing depth requirements for low-frequency variant detection", "sequencing depth"),
    ("Phylodynamics of an emerging enterovirus lineage associated with acute flaccid myelitis", "viral"),
    ("Human genome structural variants revealed by long-read sequencing", "human genome"),
    ("Random forest classification of 16S rRNA profiles in inflammatory bowel disease", "16S rRNA"),
]
JOURNALS_HI = ["Nature", "Nature Microbiology", "The Lancet", "Nature Communications", "Cell"]
JOURNALS = ["Journal of Clinical Virology", "Microbiome", "Genome Medicine", "Bioinformatics", "Clinical Infectious Diseases"]
AUTHORS = "Silva A, Okafor T, Lindgren S, Tanaka H, Rossi M, Moreau L, Chen W"


def make(i: int, source: str, journal: str) -> app.Paper:
    title, term = TITLES[i % len(TITLES)]
    rng = random.Random(i)
    abstract = (f"Background: Genomic surveillance increasingly relies on {term} approaches.\n"
                f"Methods: We analysed {100 + i} samples with nanopore and Illumina sequencing.\n"
                f"Results: The approach detected viral and bacterial pathogens with {80 + i % 15}% sensitivity "
                f"and identified carbapenemase genes on plasmids in {i % 9 + 2} isolates.\n"
                f"Conclusions: Metagenomic sequencing complements routine diagnostics.")
    return app.Paper(
        title=title if i < len(TITLES) else f"{title} ({i // len(TITLES) + 1})",
        authors=", ".join(AUTHORS.split(", ")[: 3 + i % 4]) + (" et al." if i % 4 == 3 else ""),
        abstract=abstract, url=f"https://example.org/{source}/{i}",
        published_date=date.today() - timedelta(days=rng.randint(0, 30 if journal in JOURNALS_HI else 6)),
        source=source, journal=journal, doi=f"10.1234/{source}.{i}", pmid=str(38000000 + i) if source == "pubmed" else "",
    )


def install():
    def pubmed(days, max_results, term_groups, journals, excludes=None):
        if journals:
            return [make(i, "pubmed", JOURNALS_HI[i % len(JOURNALS_HI)]) for i in range(8)]
        return [make(i, "pubmed", JOURNALS[i % len(JOURNALS)]) for i in range(8, 48)]

    async def rxiv(server, days, max_results, client):
        base = 100 if server == "biorxiv" else 200
        return [make(base + i, server, "bioRxiv" if server == "biorxiv" else "medRxiv") for i in range(15)]

    async def arxiv(days, max_results, categories, client, term_groups=None):
        return [make(300 + i, "arxiv", "arXiv") for i in range(10)]

    app._fetch_pubmed_sync = pubmed
    app.fetch_rxiv = rxiv
    app.fetch_arxiv = arxiv
    app._CACHE.clear()
