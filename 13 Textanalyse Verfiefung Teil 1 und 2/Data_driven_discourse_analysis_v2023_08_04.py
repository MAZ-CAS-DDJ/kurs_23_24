# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + [markdown] id="cS_UeDXATnbU"
# # Tutorial: Data-driven news discourse analysis with Python
#
# **August 2023**
#
# This notebook follows a Medium tutorial article, and uses Innovation Sweet Spots' public discourse analysis modules.
#
# We will fetch and analyse *The Guardian* news articles, but the analysis can also be applied to any other text data.
#
# We will provide examples for:
#
# *   Checking mentions of search terms over time
# *   Exploring the news topics using BERTopic
# *   Understanding the language used around these terms using spaCy
#
#
# For running the code locally, you can [clone the repo](https://github.com/nestauk/innovation_sweet_spots/tree/discourse_tutorial_blog) on your local machine.

# + [markdown] id="hMMXq2MuTBUn"
# ## Setting up
#
# Running the following cells will install the Innovation Sweet Spots code and other necessary python packages.
#
# Skip this step if running locally instead of Colab.

# + colab={"base_uri": "https://localhost:8080/"} id="XxjCW2Xmxs4W" outputId="d0f0a681-ea44-4f57-cb76-01d6ddd0e4c7"
# !git clone --branch discourse_tutorial_blog https://github.com/nestauk/innovation_sweet_spots.git

# + id="Up1bLlHExxX9"
import sys
sys.path.insert(0,'/content/innovation_sweet_spots')

# + colab={"base_uri": "https://localhost:8080/"} id="pLcrmaN8l968" outputId="d37b7e8a-6636-46cc-e4d7-33b3149471c5"
# !cd innovation_sweet_spots && \
# pip install -r requirements.txt

# + [markdown] id="TvoETcPFTLlv"
# ## Importing requirements

# + colab={"base_uri": "https://localhost:8080/"} id="YnB44JQfSlUK" outputId="547d8bed-e785-42c8-c938-1f6bcc1149bf"
# Import packages
import altair as alt
import pandas as pd
import nltk
nltk.download('stopwords')
from innovation_sweet_spots.innovation_sweet_spots.utils.pd import pd_analysis_utils as au

# + [markdown] id="Mrg89vXTC5ab"
# ## Getting the data: Using the Guardian Open Platform
#
# This step shows how to fetch news articles from the Guardian mentioning "heat pumps".

# + [markdown] id="k8oKTtXNC5ab"
# First you should define your Guardian API key.
#
# Setting it to `"test"` might work, but you should set up your own key here: https://open-platform.theguardian.com/access/

# + id="WwwV9QicN_v2"
API_KEY = "c69c7bae-4ebe-4423-9f6b-b82951b52358"
#https://open-platform.theguardian.com/documentation/search

# + [markdown] id="DkvYUte4C5ab"
# You can take a peek at the results by setting `only_first_page=True`
# -

import requests
import json



import urllib.parse
query = 'heat AND pump'
urllib.parse.quote(query)

# +
API_KEY = "c69c7bae-4ebe-4423-9f6b-b82951b52358"
#https://open-platform.theguardian.com/documentation/search
def search_guardian(query="test",query_fields="headline"):
    page = 1
    page_size = 50
    results = []
    total_pages = 9999
    query = urllib.parse.quote(query)
    while page <= total_pages:
        r = requests.get(f"https://content.guardianapis.com/search?api-key={API_KEY}&query-fields={query_fields}&q={query}&page-size={page_size}&page={page}")
        total_pages = json.loads(r.text)["response"]["pages"]
        print(f"Working on page {page} of {total_pages}")
        results += json.loads(r.text)["response"]["results"]
        page +=1
    return results
results = search_guardian("heat AND pump")
results += search_guardian("heat AND pumps")

[x["webTitle"] for x in results]
# -



# + colab={"base_uri": "https://localhost:8080/"} id="4Usq4o7ZC5ac" outputId="e311c938-6ce4-47ad-8f49-276985e935d8"
test_articles = au.guardian.search_content(
    "heat pumps",
    api_key=API_KEY,
    only_first_page=True,
    use_cached=True,
    save_to_cache=False
)

# + [markdown] id="-jn4GSrpC5ac"
# At the time of writing this tutorial, tt should say that 100 articles is about 14% of the total number of results, so you can work it out that there are around 700 articles on the Guardian mentioning heat pumps

# + [markdown] id="atDdg_iJC5ac"
# You can check that the most recent article

# + colab={"base_uri": "https://localhost:8080/"} id="Es0n1PAWC5ac" outputId="8b2dbb6a-d6c3-4348-8965-645a53c2a590"
# Get the first (most recent) result
test_articles[0]

# + [markdown] id="kNl2n7vXC5ac"
# Now let's get all articles mentioning heat pumps.
#
# In this example, we also specify the following article categories to reduce the possibility of irrelevant articles. All category names can be found by checking the Guardian API sections endpoint (see here an API call listing the sections).

# + id="ahbK1DWTC5ac"
# Define allowed article categories
CATEGORIES = [
    "Environment",
    "Technology",
    "Science",
    "Business",
    "Money",
    "Cities",
    "Politics",
    "Opinion",
    "UK news",
    "Life and style",
]


# + [markdown] id="qaR_sGsPTdfx"
# You can also specify multiple search terms to be included in your query. For example, in my experience, it’s best to use both singular and plural forms with Guardian API and hence we will specify both “heat pump” and “heat pumps” as our search terms here. In this implementation, each search term is queried separately and repeated hits (ie, same article featuring multiple search terms) are deduplicated.

# + colab={"base_uri": "https://localhost:8080/"} id="gqsFeW2MP73U" outputId="be743889-8acd-4143-be45-9c885d3580cd"
# List of search terms
SEARCH_TERMS = ["heat pump", "heat pumps"]

articles_df, articles_metadata = au.get_guardian_articles(
    # Specify the search terms
    search_terms=SEARCH_TERMS,
    # To fetch the most recent articles, set use_cached to False
    use_cached = True,
    # Specify the API key
    api_key=API_KEY,
    # Specify which news article categories we'll consider
    allowed_categories = CATEGORIES,
)


# + colab={"base_uri": "https://localhost:8080/", "height": 143} id="gHrW-OCSOHNd" outputId="77bab177-2638-40c5-f66b-16effdd5ab8b"
# Article texts
articles_df.head(3)

# + colab={"base_uri": "https://localhost:8080/"} id="LsdixR-HC5ac" outputId="13c1c195-bda9-4bee-8f27-9f9f41f1ed75"
# Article metadata
articles_metadata[articles_df.iloc[0].id]

# + [markdown] id="SLmocSRAfK9h"
# ## Initialising the `DiscourseAnalysis` class
#
# First, we can specify the path to the analysis outputs directory, which will come handy when revisiting the analysis in the future. Note that we are storing the analysis outputs separately from the cached search results (discussed above), in order to separate the analysis process, which is agnostic to the data sources, from the data fetching process

# + id="LMOcW3v7C5ac"
# Specify the location for analysis outputs
from innovation_sweet_spots import PROJECT_DIR
OUTPUTS_DIR = PROJECT_DIR / "outputs/data/discourse_analysis_outputs"

# + [markdown] id="FM_gNMwTC5ac"
# We can then specify the name `ANALYSIS_ID` for this specific analysis session - all the output tables will be stored in a subfolder of `OUTPUTS_DIR` with the same name.

# + id="SCrWQ0bWC5ac"
ANALYSIS_ID = "guardian_heat_pumps_tutorial"

# + [markdown] id="hksmGeaBC5ac"
# We will be saving and loading our analysis results to and from `innovation_sweet_spots/outputs/data/discourse_analysis_outputs/{ANALYSIS_ID}`.
#
# We will then define a couple of additional filtering criteria to keep the most relevant results to our context, by specifying a (non-exhaustive) list of UK-related geographic terms and excluding any article that mentions Australia.

# + id="FxR1xGLVC5ac"
# Terms required to appear in the articles,
# for the articles to be considered in the analysis
REQUIRED_TERMS = [
    "UK",
    "Britain",
    "Scotland",
    "Wales",
    "England",
    "Northern Ireland",
    "Britons",
    "London",
]

# Articles with these terms will be removed from the analysis
BANNED_TERMS = ["Australia"]

# + colab={"base_uri": "https://localhost:8080/"} id="6F76PoYgTFkX" outputId="f7e40470-7205-4ca4-8692-dc61d0b6cd72"
pda = au.DiscourseAnalysis(
    search_terms=SEARCH_TERMS,
    outputs_path=OUTPUTS_DIR,
    query_identifier=ANALYSIS_ID,
    required_terms = REQUIRED_TERMS,
    banned_terms = BANNED_TERMS,
)

pda.load_documents(document_text=articles_df)

# + [markdown] id="41Qnn4UQgF65"
# The warning message above says we are missing document text and metadata. Metadata is optional and can be used when using *Guardian* articles.
#
# The `load_documents` step adds document text to the class. This function has an argument `document_text` which can take a dataframe variable or if left blank will search for a file `document_text_{ANALYSIS_ID}.csv` in `outputs/data/discourse_analysis_outputs/{ANALYSIS_ID}/`.
#
# Note that you can use `load_documents` to input any text data, as long as it has columns for `text`, `date`, `year` and `id`.

# + [markdown] id="kR9kRkr7jIFM"
# ## Number of news articles across years

# + [markdown] id="ClGzXutg58ct"
# The number of documents per year that contain the search terms.
#
# (The results for each search term are combined and deduplicated)

# + colab={"base_uri": "https://localhost:8080/", "height": 802} id="ZwlsVrCMsrkd" outputId="f93b475e-7da4-4a8c-fe0d-9e940cefc4dc"
pda.document_mentions

# + [markdown] id="Ghh8g7T06DxT"
# Plot of the number of documents per year that contain the search terms.

# + colab={"base_uri": "https://localhost:8080/", "height": 381} id="XTPG_WSYNQPX" outputId="eeb0a595-29b5-4270-9411-06b3515e4c18"
pda.plot_mentions(use_documents=True)

# + [markdown] id="8OKThKA59gj3"
# You can also plot the number of sentences, and disaggregate the number of sentences per each search term.
#
# (This might take a minute, as the text is processed into sentences using spacy)

# + colab={"base_uri": "https://localhost:8080/", "height": 376} id="cgmsBXD7sk6g" outputId="ecab4600-1200-4a30-f1aa-46489f7b82c8"
pda.plot_mentions(use_documents=False)

# + [markdown] id="tz8XimxT9kYB"
# You can then get all sentences with the search terms for a specific year, using the dictionary `combined_term_sentences`

# + colab={"base_uri": "https://localhost:8080/", "height": 372} id="DNC-N8Dms86O" outputId="b136acfd-c515-45ad-c018-1b9de15fef2e"
pd.set_option('max_colwidth', 500)
pda.combined_term_sentences["2022"].head(5)

# + [markdown] id="r62Poj0qC5ad"
# Finally, when considering the growth trends of news mentions, another important element is a baseline growth trend that we can use as a reference.

# + id="BwJuRkk1C5ad"
# Get the total article counts across relevant article categories
total_counts = au.get_total_article_counts(sections=CATEGORIES, api_key=API_KEY)

# + [markdown] id="lhbuWtXuC5aj"
# After dividing the number of articles mentioning heat pumps with the total number of reference articles, we find that the shape of the trend is preserved.

# + colab={"base_uri": "https://localhost:8080/", "height": 381} id="Co4XhLy8C5aj" outputId="ba2e532c-3bdc-43a1-803a-6c398dff40cf"
document_mentions_norm = (
    pda.document_mentions.copy()
    .assign(baseline_documents = total_counts.values())
    .assign(normalised = lambda df: df.documents / df.baseline_documents)
)

alt.Chart(document_mentions_norm).mark_line().encode(x="year:O", y="normalised")

# + [markdown] id="HdkcKIYlvgf1"
# ## Characterising discourse topics using BERTopic
#
# We can use BERTopic to find topics within our documents. More info on BERTopic can be found [here](https://maartengr.github.io/BERTopic/faq.html).
#
# To create a topic model, use function `fit_topic_model`. If you want to use sentences found from phrase matching set the variable `use_phrases` to `True` (note, if using phrases, `set_phrase_patterns` will need to be run first.) If set to `False` it will use the `sentence_mentions`.

# + colab={"base_uri": "https://localhost:8080/", "height": 521, "referenced_widgets": ["1d006720a66e431e9de2fc8d2d92fe7a", "f9d4fe8c6aca4edfb8264d2af8e66b24", "ef187fa1814247a59f3499dc5a1b2f2c", "5956e5b8e7c54938a19fbbabba11598a", "8ab1946b31e742069267384dfe3c7ba5", "dea3d56fc73d4ac28411f1767f723af3", "d1e071be21694982b257f816a483b314", "518c5957c295406488e83f51cdb84552", "cf8903ce35cd4815bef4df37e8b8eb74", "7d68a084a068492ca0522b450fb718b6", "bb727e58e9e144078cafdb46146da762", "b15e2c15e1004c1e8b2391cbc87ab680", "e491264e4407432fbd396b71a3c5ab15", "f568ff97858c4a4db53c87b7ee0304d8", "a5156d32997049e0bb3153a21a1c250f", "56ac317be250422cb53009e13ca600ab", "d3d56339c29b4d96903a987074b21634", "c3360a90c22e4f65bb8370d6c6b9681c", "39e1ffd5ca224001a78a933a55d5a5be", "783cbcf70c6b4646949544153e19e681", "531b725b0b2c4c0b9293d17a0a9f85cc", "95a1fe91a5d84c71996b97c76cf416c6", "4c75aaf7f2cf469baac10ee0e6cf202b", "888373094d2943518773aba191f23d28", "da9e244193b6431fa05a0047ec4a9f29", "51f5ed32c93d4e129b8f76b5da3d4017", "75c96a7563d94734ab8ee28b248f4642", "80dbca3d66d34a12aafcea878f01975e", "3d3086265ef640c7bbd4c07982796719", "7576ff3c4ef54b349a98d928834826ba", "6023c07666ae4f43bdb6c71b7e1ec576", "3a1c87ca0e2a4262bfac5e40db063957", "9c1341f9c17346e78e3de9a604b6d0b9", "47bd849b56074da9a82e7e4f051af089", "6bb0a267d90a4ac185a419de52d63e68", "c9ccf8ce8c0545a79fc849669f3acdc3", "1d52a31a66c0486bb8194f4b62227aa5", "0046a34f89b94408a12ef4dd27288e22", "67606306a15642ffbe0272b54c01f46b", "7f9c76dc503a4266a798f98e6a881b31", "78b9471df8934ddc998f4aaa11cc69fc", "7262705e81124c81aea2729129cc9aab", "79be4a47c54e4681a31ee1783ae63236", "115a5c6ca7184a64b04ba5167a2468fb", "6965dd62c05b4e2b8ad109ac7fbdf7b9", "85c61385b51d420eb177038792694d59", "abd5b3ff8dfd45158154204bcd452be1", "76d8a34fff0f4a839cb81c7518983970", "e32e3658404247aba3aa68e14a3a62a0", "bf974a3742d94f22935b2adbf80f72ae", "f5661043e04143c198a8e66e41d5ec06", "d751f4d4073b45dd9de16cdcce1f7c42", "8d484a950d8a4a62990a92e2ddeaffa1", "6fcaf747cb334cc6baca252837ee3d42", "a1ca51dfa5b64bfe8b84b1ddf960ecbe", "c45c9c9d5fa8480285d9a9c57d19f019", "a3366a4caf3a4afda10ad7abe7c08342", "70b621cfeb5e4d8298b0f1be60fec62d", "f5687432477e4d4293ae75a604dea8fe", "31b2aa8808044b48ad5a731629663fcc", "1b5a235921c241e98cd8967b6a14db51", "656cc28caa044c5b91da4ecb8bc80435", "50fc274837554dd1994ee9a62505151c", "3e4a9a0d5f7e4e5cb96a5fd144731f68", "f5391581b85c494cabed746a4d56a77e", "58823da1fb5b4d0192fe84cfe08174ac", "955c8396a7b64b1ea5aa469ef849486d", "c0a0746d520d43a3b20785714e229ab3", "c8c2a109caed412a98a4d9f19496d750", "faa44e1a72db439793f28c8aed1ed99a", "66dec27de430462c98422a5067499fb8", "bf0be29ed05d4b58bee737941f556998", "d232a10ba0854e428fd863dac60ae5c9", "6e24544dde7e4f1db53944b7ee77731d", "2e961d82e5bf44c4a7cfa48f16a3eaa6", "784eb5c9333f47b096d606b60cc9bcf6", "64fa2f87d9564fdd8e2c9c436cc59c69", "ff823818283a4543bb9f90eee061cb3b", "f2cd44baf6c04112917d2b93405ef42d", "4caed2b9c9c44f858ce1899c6d393721", "762f670052574c46ac3299b41a86bc9c", "5d734583733f4cd9a3edb5fabcf36534", "8ebd89518b6a4a7496aa5bbf183da063", "e4a5de5f9cfa4039ba6e5a0eee8db33e", "f385063ab94d40ecb3949175c74d0148", "c2ae3fe56dbf4584b1963953dc951eae", "6de53c41c5524b0a9679315e67b05d97", "8db4e6848e96432f85a07d66831d1ef2", "14f0f0d9aec2437484ec2cb5c14e4f7f", "eabffd97060a428aa724eefe3ba36bef", "bf40ede9abe3442bbf3b4b30a5b98fbd", "b8cf6d54e6e94c0f96d9d35a0722e38c", "d342e88aa49149d3a08f753996b8e57e", "aaee759dadaf47c89d42bb76fe2e6d28", "f565d0ce2f444ba39c00538668266866", "ce78544c4f3b4b938e48904258154fb7", "c9b1ec93f0d44840aaf7675ed1022a76", "6bebcf5d75374daebd5539b8b10c9b52", "8a9ec0571e6f4cf694119da72fd13965", "ff063a7cdb2c4d58ad21b86edb957730", "0d35619fdccc4ed5b052985855d05365", "b346dead5c244837bec9b6833f658da3", "b4f37fded8d548e7ad1ec481f09c0165", "3c7542c040104c37afcaffc753febae9", "eceee2690f22425db74d02eaafd1d502", "8debf55a4ef8464691b4dd82f56e4d77", "9352e2dc5d644672bb6aa8e77b466764", "9263d4d4a69f45b4abac33e59c046f9b", "3ba98c12d44f44a5ae7c47f495041ac0", "4c21c05576ec4442b643405330f83e35", "631ebe9ecee849c5bc6ddf910988347a", "b19072a3e56848d888002dd1ae02c5f0", "d9a679e4ad1343279cb812c0089874d6", "de28006622c946d89304958d09bdc1eb", "32f3b2e658d54b1ca1df764628fd9eaa", "6e353855fd2749ca963c0863068982aa", "414f1d7fb6fb4d01b854344e81e4a48e", "1a97e48130964a33a9a3a6902f4c2227", "e38cfad158d84719942868111132c412", "f353187dd9fc4c97929637217d072b64", "312ab9071743424db2f832bc3e8f51b5", "36418f32edea42f48f4d223099a61833", "4a0a95165f1740d9a5ed2a5f802a5c9b", "9f360e9a063b489b9a64339eb6b13f12", "c636e3b1f23c4298a122451d139b1764", "900b9d1770d94c4a8fea510121dd2bdd", "a3b24d0f304c4845b57dd1204204d99e", "cbfdbc8a0a4b49eb944c64f1ffbf9604", "61d997ff37a04dc0a3a352b2543ce5cf", "942247935e7143f9a2c9f7d435219eff", "cdebd03935e34f46974c8ada77a32b67", "807f7831c6204332ba3a8ee6d28f8e3c", "e7339f9303db4365a1545241e307047f", "4fd4916d7ee94f7e8eca32c57f775fe9", "863f8325a3594c759751c3b3274cebd8", "53ff17d74cc64f46822e1b3f9ecf6d6e", "2b6d6f8a8dce42a3bbc11ee8721144b4", "6fdc61bc5d6b41e8bb6c9060657c058d", "0885eb9be0b64459ba99792583246bec", "20b4dff7454a42419cc9e0707667a8c8", "78436ea870aa4608bd7db9f463815351", "41423b7385d942bcb79ca6c841f991d7", "a23892c6e5fe4725ba417eb1dc3442d4", "e253179cc3364868977c2fb0d87195c3", "d7a6e9e9cb5b4ebb86099065e70a107e", "497ae4ae31834879be8d893d5352c320", "6fa42cf85b664a9cae5988a6b6b6e3f4", "c6d3cd423b8f44e29fde667211d5e5d7", "f0f8e5877d19436681d519dbe725c662", "28bb3396d92d4e11ac0458fa3e65b46d", "9e61ab0d44164b9f812c2992cd30028f", "4117bf400b6a495bbf01c02f5f8979ad", "65f72483ee4b4653b5630ad680a26a89", "f638067d18014f2394b93525805dccb8"]} id="sntosuXWC5ak" outputId="11149108-e744-4f51-b9a3-9e3f46a3bce3"
topic_model, docs = pda.fit_topic_model()

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="FesQDN4QC5ak" outputId="f8df449a-6008-4c03-90e5-40adbb982e7e"
topic_model.visualize_barchart(top_n_topics=len(set(topic_model.topics_)))

# + [markdown] id="A0LFG8o2C5ak"
# Visualising the documents and topics in 2d space (mouseover for the text). Note that the `visualize_documents` function is not determinstic. You can make the output deterministic by following the steps [here](https://maartengr.github.io/BERTopic/faq.html).
#
# To make the plot larger (to be able to view more of the mouse over text, increase the values for the parameters `width` and `height`)
#
# Google Colab might have issues with plotly plots. In that case you can try downloading the chart as an html.

# + id="WHovaH25C5ak"
fig = topic_model.visualize_documents(docs, width=800, height=500)

# + colab={"base_uri": "https://localhost:8080/", "height": 517} id="euNB3ACM1QiO" outputId="d91c77cd-abec-4fd1-9cf4-01bd3f52394d"
fig

# + id="6pvxjyMi1UJv"
fig.write_html('document_topic_chart.html')

# + [markdown] id="3iyCpbEZvElX"
# ## What we talk about when we talk about X: Collocation analysis
#
# The `view_collocations` function can be used to find sentences where the search term appears with another specified term.

# + colab={"base_uri": "https://localhost:8080/", "height": 711} id="bANvuhdFvWVE" outputId="845e756c-0595-434b-eec9-b08f30bae3a5"
pda.view_collocations("air source")

# + [markdown] id="vq9fqYSmC5ak"
# You can check `term_rank` table for all collocated terms for each year
#
# The importance of collocations is measured by pointwise mutual information (pmi), which indicates the strength of association of the collocated term and search term.
#
# Frequency (freq) and rank indicate how often the terms have been used together with the search terms. Frequency is the number of co-occurrences.
#
# Note that this might take a minute. As an alternative, you can also ran a simpler query `analyse_colocated_unigrams` to find frequently mentioned single-word terms.

# + colab={"base_uri": "https://localhost:8080/", "height": 424} id="D54f227WC5ak" outputId="0a5d0f76-b92a-49c9-a35e-da2bb9dbc7a8"
pda.term_rank

# + colab={"base_uri": "https://localhost:8080/", "height": 708} id="nB67ux_vC5ak" outputId="69a66d46-1d68-4de9-f1b0-e68f2c2d4a69"
# Check most often collocated terms
(
    pda.term_rank.groupby('term')
    .agg(freq=('freq', 'sum'))
    .sort_values('freq', ascending=False)
    .head(20)
)

# + colab={"base_uri": "https://localhost:8080/", "height": 677} id="Lcqc41Z0C5ak" outputId="2e8327ed-1613-41c4-f3d1-56db27022561"
# Check collocations with highest PMI in 2021
(
    pda.term_rank
    .query("year == 2021")
    .sort_values('pmi', ascending=False)
).head(20)

# + [markdown] id="lt3cLUPPC5ak"
# Try out also this simpler approach using only unigrams (will work on Colab)

# + colab={"base_uri": "https://localhost:8080/", "height": 677} id="8TJYsplnC5ak" outputId="d45e2963-e746-4eb9-cb7f-304b8e78c906"
# Simpler measure using only unigrams
pda.analyse_colocated_unigrams().sort_values('counts', ascending=False).head(20)

# + [markdown] id="EEs7MRQjC5ak"
# ## Shifting narratives: Collocation importance over time
#
# It is also informative to analyse how the importance of collocated terms varies over time. This can point to shifts in language used to describe our search terms, which in our case might be caused by the emergence of new entities and applications associated with our technologies of interest.
#
# For example, a comparison of the two different types of heat pumps discussed above highlights an interesting trend, where ‘ground source’ was initially more frequently collocated, whereas now ‘air source’ - which is a more affordable type of heat pump - has overtaken the mentions of ‘ground source’.
#

# + id="Yo9CLofFqdEB"
# Define a helper function
from itertools import product

def create_time_series_df(check_terms: list, pda: au.DiscourseAnalysis):
    """Helper function to create a time series table"""
    # Get frequencies for specific terms
    terms_df = (
        pda.term_rank.query("term in @check_terms")
    )
    # Create a time series table for each term and concatenate them
    time_series_df = []
    for term in check_terms:
        time_series_df.append(
            # Create a table with all years
            pd.DataFrame(list(product(list(range(terms_df.year.min(), terms_df.year.max() + 1)), [term])), columns=["year", "term"])
            # Merge with the term frequencies
            .merge(terms_df[["year", "term", "freq"]], on=["year", "term"] ,how="left")
            .fillna(0)
            # Create a rolling mean
            .assign(freq_ma3y=lambda x: x.freq.rolling(3, center=True, min_periods=1).mean())
        )
    return pd.concat(time_series_df, ignore_index=True)



# + colab={"base_uri": "https://localhost:8080/", "height": 366} id="Nlrn17IIC5ak" outputId="2270c039-a320-4fe2-9414-b692d16932f2"
# Define collocated terms to check
check_terms = ['air source', 'ground source']
# Call helper function (defined above) to create a (smoothed) time series
time_series_df = create_time_series_df(check_terms, pda)

fig = (
    alt.Chart(time_series_df)
    .mark_line(interpolate="monotone")
    .encode(
        x=alt.X('year:O', title=''),
        y=alt.Y('freq_ma3y:Q', title='Frequency (moving average)'),
        color=alt.Color('term:N', title='Term'),
    )
)
fig


# + [markdown] id="7Ymv5_kZC5ak"
# The `term_temporal_rank` can be used to potentially highlight interesting terms whose rank has changed significantly (ie, has a high variation across years)

# + colab={"base_uri": "https://localhost:8080/", "height": 865} id="kWxNaqWkC5ak" outputId="4708a594-2976-4c4e-a68a-6b40f5ab065c"
pda.term_temporal_rank.sort_values('st_dev_rank', ascending=False).head(25)

# + [markdown] id="zQVgkHIcvarM"
# ## Extracting patterns using spaCy
#
# We can also analyse specific types of phrase patterns containing the search term. For example, phrases that contain adjectives or verbs together with the search term.
#
# More information of POS phrase matching can be found [here](https://spacy.io/usage/rule-based-matching).
#
# The phrase patterns can either be loaded from a json file  theor default phrase patterns can be generated on the fly using `make_patterns=True`.

# + [markdown] id="pb6dXTNTUnPl"
# First we need to set the phrases patterns. Here we are making patterns based on the search terms.

# + colab={"base_uri": "https://localhost:8080/"} id="RyzjXTKHve4-" outputId="99c53a21-ce53-455c-8a8d-e3994bbe13ae"
pda.set_phrase_patterns(load_patterns=False, make_patterns=True)

# + colab={"base_uri": "https://localhost:8080/"} id="rmnq2dvHC5al" outputId="e6581dd7-1fef-490e-81eb-fe363f848ab4"
pda.set_phrase_patterns(load_patterns=False, make_patterns=True).keys()

# + [markdown] id="KaBnGPJGUryd"
# Then we can find matches in the documents that match the phrase patterns.
#
# This might take a minute.

# + colab={"base_uri": "https://localhost:8080/", "height": 497} id="BzhWqijaC5al" outputId="cffced99-6ba1-411f-aad0-57617ef13255"
pda.pos_phrases

# + colab={"base_uri": "https://localhost:8080/"} id="nWHUSIMxC5al" outputId="4e538bf1-b556-4b63-f51d-c5c94500768e"
# All types of patterns generated
# (note that we have separate patterns for singular and plural search terms)
sorted(pda.pos_phrases.pattern.unique())

# + colab={"base_uri": "https://localhost:8080/", "height": 1000} id="SQ4ZfNc3WLXi" outputId="cc283c37-5448-449c-8070-3b58d48820dc"
# Query the adjective phrase patterns
pos = 'adj_phrase'
(
    pda.pos_phrases
    .groupby(['phrase', 'pattern'], as_index=False)
    .agg(number_of_mentions=('number_of_mentions','sum'))
    .query(f"pattern.str.contains('{pos}')", engine='python')
    .sort_values('number_of_mentions', ascending=False)
    # .head(10)
)

# + [markdown] id="M4_-cWOpC5al"
# ## Save the analysis outputs
#
# Speeds up the analysis. Next time, if you specify the same query id, it should load the results and you won't have to compute the phrases and patterns again.

# + id="e-3GKstwwJTH"
pda.save_analysis_results()
