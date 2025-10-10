from LLM import complete_text, complete_text_claude
import arxiv
import scholarly
from pymed import PubMed
from urllib import request
from bs4 import BeautifulSoup
import time
import random

DEFAULT_URL = {
    'biorxiv':
    'https://www.biorxiv.org/search/{}%20numresults%3A25%20sort%3Arelevance-rank'
}

class MyBiorxivRetriever():
    def __init__(self, search_engine='biorxiv', search_url=None):
        self.search_engine = search_engine
        self.search_url = search_url or DEFAULT_URL[search_engine]
        return

    def _get_article_content(self,
                             page_soup,
                             exclude=[
                                 'abstract', 'ack', 'fn-group', 'ref-list'
                             ]):
        article = page_soup.find("div", {'class': 'article'})
        article_txt = ""
        if article is not None:
            for section in article.children:
                if section.has_attr('class') and any(
                        [ex in section.get('class') for ex in exclude]):
                    continue
                article_txt += section.get_text(' ')

        return article_txt

    def _get_all_links(self, page_soup, max_number, base_url="https://www.biorxiv.org"):
        links = []
        for link in page_soup.find_all(
                "a", {"class": "highwire-cite-linked-title"})[:max_number]:
            uri = link.get('href')
            links.append({'title': link.text, 'biorxiv_url': base_url + uri})

        return links

    def _get_papers_list_biorxiv(self, query, max_number):
        papers = []
        url = self.search_url.format(query)
        page_html = request.urlopen(url).read().decode("utf-8")
        page_soup = BeautifulSoup(page_html, "lxml")
        links = self._get_all_links(page_soup, max_number)
        papers.extend(links)
        return papers
    
    def query_short(self, query, max_number, metadata=True, full_text=True):
        query = query.replace(' ', '%20')

        if self.search_engine == 'biorxiv':
            papers = self._get_papers_list_biorxiv(query, max_number)
        else:
            raise Exception('None implemeted search engine: {}'.format(
                self.search_engine))

        return papers

    def query_entire_papers(self, papers):

        for paper in papers:
            biorxiv_url = paper['biorxiv_url'] + '.full'
            page_html = request.urlopen(biorxiv_url).read().decode("utf-8")
            page_soup = BeautifulSoup(page_html, "lxml")

            abstract = page_soup.find("div", {
                'class': 'abstract'
            })
            if abstract is not None:
                paper['abstract'] = abstract.get_text(' ')
            else:
                paper['abstract'] = ''

            article_txt = self._get_article_content(page_soup)
            paper['full_text'] = article_txt

        return papers


def understand_file(lines, things_to_look_for, model):

    blocks = ["".join(lines[i:i+2000]) for i in range(0, len(lines), 2000)]

    descriptions  = []
    for idx, b in enumerate(blocks):
        start_line_number = 2000*idx+1
        end_line_number = 2000*idx+1 + len(b.split("\n"))
        prompt = f"""Given this (partial) file from line {start_line_number} to line {end_line_number}: 

``` 
{b}
```

Here is a detailed description on what to look for and what should returned: {things_to_look_for}

The description should short and also reference crtical lines in the script relevant to what is being looked for. Only describe what is objectively confirmed by the file content. Do not include guessed numbers. If you cannot find the answer to certain parts of the request, you should say "In this segment, I cannot find ...".
"""

        # Use appropriate completion function based on model
        if 'claude' in model.lower():
            completion = complete_text_claude(prompt, model=model, log_file=None)
        else:
            completion = complete_text(prompt, model=model, log_file=None)
            
        descriptions.append(completion)
        
    if len(descriptions) == 1:
        return descriptions[0]
    else:
        descriptions = "\n\n".join([f"Segment {idx}: \n\n" + s for idx, s in enumerate(descriptions)])
        prompt = f"""Given the relevant observations for each segments of a file, summarize to get a cohesive description of the entire file on what to look for and what should returned: {things_to_look_for}

{descriptions}
"""

        # Use appropriate completion function based on model
        if 'claude' in model.lower():
            completion = complete_text_claude(prompt, model=model, log_file=None)
        else:
            completion = complete_text(prompt, model=model, log_file=None)

        return completion


def is_query_similar(new_query, used_queries, threshold=0.6):
    """Check if new query is too similar to previously used ones"""
    from difflib import SequenceMatcher
    
    for used_query in used_queries:
        similarity = SequenceMatcher(None, new_query.lower(), used_query.lower()).ratio()
        if similarity > threshold:
            return True
    return False


def what_to_query(current_prompt, model, used_queries=None):
    """
    Generate a focused query, avoiding repetition of previous queries
    """
    if used_queries is None:
        used_queries = []
    
    # Add instruction to avoid repetition
    avoid_instruction = ""
    if used_queries:
        avoid_instruction = f"\n\nAvoid generating queries too similar to these previously used ones: {', '.join(used_queries[-3:])}"
    
    prompt = f'''
You are an expert at literature review. You are given the current state of the research problem and some previously done research: 

{current_prompt}

Your task is to come up with a one-line very focused query without any additional terms surrounding it to search relevant papers which you think would help the most in making progress on the provided research problem.

Focus on different aspects like: gene prioritization strategies, computational approaches for CRISPR screen optimization, interferon-gamma pathway regulation, experimental design for large-scale screens, etc.{avoid_instruction}
    '''
    
    print("Current prompt:", current_prompt[:200] + "..." if len(current_prompt) > 200 else current_prompt)
    
    # Use appropriate completion function based on model
    if 'claude' in model.lower():
        query = complete_text_claude(prompt=prompt, model=model, log_file='paper_search.log')
    else:
        query = complete_text(prompt=prompt, model=model, log_file='paper_search.log')
    
    return query.strip()


def query_pubmed_with_retry(pubmed, query, max_results=4, max_retries=3):
    """Query PubMed with retry logic and proper error handling"""
    
    for attempt in range(max_retries):
        try:
            print(f"Attempting PubMed query (attempt {attempt + 1}): {query}")
            papers = list(pubmed.query(query, max_results=max_results))
            return papers
        except KeyboardInterrupt:
            print("Query interrupted by user")
            raise
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with error: {e}")
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                print(f"Waiting {wait_time:.1f} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"All {max_retries} attempts failed for query: {query}")
                return []
    
    return []


def get_lit_review(prompt, model, max_number=4, max_queries=3):
    """
    Get literature review with proper loop control and error handling
    
    COMPLETELY REWRITTEN to prevent infinite loops and repetition
    """
    
    
    print(f"Max queries limit: {max_queries}")
    
    lit_review = ""
    used_queries = []
    successful_queries = 0
    
    # Initialize PubMed client
    timer = random.randint(0, 1e6)
    pubmed = PubMed(tool="MyTool", email=f"{timer}@email.address")
    pubmed._rateLimit = 11
    
    print(f"Starting literature review with max {max_queries} queries...")
    
    # FIXED LOOP: No more infinite while True!
    for query_attempt in range(max_queries):
        try:
            print(f"\n--- Starting Query Round {query_attempt + 1}/{max_queries} ---")
            
            # Generate new query with context of used queries
            query = what_to_query(prompt, model, used_queries)
            
            # Check for similarity to avoid repetition
            if is_query_similar(query, used_queries):
                print(f"Query too similar to previous ones, generating alternative...")
                # Try to generate a different query by modifying the prompt
                modified_prompt = prompt + f"\n\nGenerate a query focusing on a different aspect than: {', '.join(used_queries)}"
                query = what_to_query(modified_prompt, model, used_queries)
                
                # If still similar, force a different approach
                if is_query_similar(query, used_queries):
                    print("Still too similar, using predefined alternative...")
                    alternative_queries = [
                        "machine learning gene prioritization CRISPR screens",
                        "transcriptional regulators interferon gamma signaling",
                        "adaptive experimental design genome-wide perturbations",
                        "JAK-STAT pathway genetic screen optimization",
                        "cytokine production regulatory networks CRISPR"
                    ]
                    query = alternative_queries[query_attempt % len(alternative_queries)]
            
            # Add to used queries BEFORE using it
            used_queries.append(query)
            print(f"Final query {query_attempt + 1}: {query}")
            
            # Query PubMed with retry logic
            papers = query_pubmed_with_retry(pubmed, query, max_results=max_number)
            
            if not papers:
                print(f"No papers found for query: {query}")
                continue
            
            # Process papers
            query_review = f"\n{'='*50}\nQuery {query_attempt + 1}: {query}\n{'='*50}\n"
            
            for i, paper in enumerate(papers):
                try:
                    query_review += f'\n--- Paper {i+1}: {paper.title} ---\n'
                    
                    # Compile paper information
                    prompt_for_summary = ""
                    if paper.title:
                        prompt_for_summary += f"Title: {paper.title}\n"
                    if paper.abstract:
                        prompt_for_summary += f"Abstract: {paper.abstract}\n"
                    if hasattr(paper, 'methods') and paper.methods:
                        prompt_for_summary += f"Methods: {paper.methods}\n"
                    if hasattr(paper, 'conclusions') and paper.conclusions:
                        prompt_for_summary += f"Conclusions: {paper.conclusions}\n"
                    if hasattr(paper, 'results') and paper.results:
                        prompt_for_summary += f"Results: {paper.results}\n"
                    
                    if prompt_for_summary.strip():
                        summarized_paper = understand_file(
                            prompt_for_summary, 
                            f"Information about genes, pathways, experimental methods, or insights relevant to CRISPR screening for interferon-gamma regulation. Focus on: {prompt}", 
                            model
                        )
                        query_review += f"{summarized_paper}\n\n"
                    else:
                        query_review += "No content available for this paper.\n\n"
                        
                except Exception as e:
                    print(f"Error processing paper {i+1}: {e}")
                    query_review += f"Error processing this paper: {e}\n\n"
            
            lit_review += query_review
            successful_queries += 1
            
            print(f"Successfully processed query {query_attempt + 1}")
            
        except KeyboardInterrupt:
            print("Literature review interrupted by user")
            break
        except Exception as e:
            print(f"Error in query {query_attempt + 1}: {e}")
            continue
    
    print(f"Literature review completed. Processed {successful_queries} queries successfully.")
    
    if not lit_review.strip():
        lit_review = "No literature review content was successfully retrieved."
    
    return lit_review


# Keep your existing search functions
def biorxiv_search(query, max_number, folder_name=".", **kwargs):
    br = MyBiorxivRetriever()
    papers = br.query_short(query, max_number)
    papers_full = br.query_entire_papers(papers)
    return papers


def arxiv_search(query, max_papers, folder_name=".", **kwargs):
    client = arxiv.Client()

    search = arxiv.Search(
        query=query,
        id_list=[],
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
    )
    
    observation = ""
    for paper in client.results(search):
        observation += "\n" + paper.title + "\n\n" + paper.summary + "\n"

    return observation


def scholar_search(query, max_papers, folder_name=".", **kwargs):
    search_query = scholarly.search_pubs(query)
    scholarly.pprint(next(search_query))

    search = arxiv.Search(
        query=query,
        id_list=[],
        max_results=max_papers,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending
    )
    
    observation = ""
    for result in arxiv.Client().results(search):
        observation += "\n" + result.title + "\n\n" + result.summary + "\n"

    return observation


def paperqa_search(query, max_papers, folder='.', **kwargs):
    from paper_scrapper import paper_scrapper
    import paperqa
    
    papers = paper_scrapper.search_papers(query, limit=max_papers)
    docs = paperqa.Docs()
    for path, data in papers.items():
        try:
            docs.add(path)
        except ValueError as e:
            print('Could not read', path, e)

    answer = docs.query(query)
    return answer