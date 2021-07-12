import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
import re
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances


def most_similar(doc_id,similarity_matrix,matrix):
    
    print ('\n')
    print ('Similar Documents:')
    if matrix=='Cosine Similarity':
        similar_ix=np.argsort(similarity_matrix[doc_id])[::-1]
    elif matrix=='Euclidean Distance':
        similar_ix=np.argsort(similarity_matrix[doc_id])
    for ix in similar_ix:
        if ix==doc_id:
            continue
        print('\n')
        print (f'Document %d:'%(ix+1))
        print (f'{matrix} : {similarity_matrix[doc_id][ix]}')
doc1_path="./document1.txt"
doc2_path="./document2.txt"
doc3_path="./document3.txt"
doc4_path="./document4.txt"
doc5_path="./document5.txt"
doc6_path="./document6.txt"
print("------------Calcul de similarité par Cosine Similarity---------------")
for j in range(6):
 print("---------------------La similitude de document %d avec les autres documents ---------------------"%(j+1))
 doc_path=[doc1_path,doc2_path,doc3_path,doc4_path,doc5_path,doc6_path]
 doc_data=[]
 for i in range(len(doc_path)):
  with open(doc_path[i], 'r') as doc: 
    doc_data.append(doc.read())  
 # Sample corpus
 documents_df=pd.DataFrame(doc_data,columns=['documents'])
 # removing special characters and stop words from the text
 stop_words_l=stopwords.words('french')
 documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
 tfidfvectoriser=TfidfVectorizer()
 tfidfvectoriser.fit(documents_df.documents_cleaned)
 tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)
 pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
 most_similar(j,pairwise_similarities,'Cosine Similarity')
 
 
 
print("------------Calcul de similarité par la distance euclidienne---------------")

 
for j in range(6):
 print("---------------------La similitude de document %d avec les autres documents ---------------------"%(j+1))
 doc_path1=[doc1_path,doc2_path,doc3_path,doc4_path,doc5_path,doc6_path]
 doc_data1=[]
 for i in range(len(doc_path)):
  with open(doc_path1[i], 'r') as doc: 
    doc_data1.append(doc.read())  
 # Sample corpus
 documents_df=pd.DataFrame(doc_data1,columns=['documents'])
 # removing special characters and stop words from the text
 stop_words_l=stopwords.words('french')
 documents_df['documents_cleaned']=documents_df.documents.apply(lambda x: " ".join(re.sub(r'[^a-zA-Z]',' ',w).lower() for w in x.split() if re.sub(r'[^a-zA-Z]',' ',w).lower() not in stop_words_l) )
 tfidfvectoriser=TfidfVectorizer()
 tfidfvectoriser.fit(documents_df.documents_cleaned)
 tfidf_vectors=tfidfvectoriser.transform(documents_df.documents_cleaned)
 pairwise_similarities=np.dot(tfidf_vectors,tfidf_vectors.T).toarray()
 pairwise_differences=euclidean_distances(tfidf_vectors)
 most_similar(j,pairwise_differences,'Euclidean Distance') 
 




 

 

