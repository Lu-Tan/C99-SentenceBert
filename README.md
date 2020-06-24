# C99-SentenceBert
For linear text segmentation

This algorithm is from Freddy Y.Y.Choi in his paper *Advances in domain independent linear text segmentation* in 2000. (really too old). I feel it strange that no implementation of C99 algorithm is found to the best of my knowledge. I can only find an api in Java. But due to class sun.misc.Compare was deleted from java since java 8, the api didn't work.

In this repo, I implemented C99 algorithm. 

1. Load sentence-bert model from URL and calculate embeddings of the whole document.
2. Calculate cosine similarities of each sentences and put them in the matrix: sim_matrix
3. Since an additional occurence of a common word causes a disproportionate increase in sim_matrix in short text segments, we present a ranking scheme which is an adaptation. Each value in the similarity matrix is replaced by its rank in the local region. The rank is the number of neighbouring elements with a lower similarity value.
  An example of image ranking using a 3x3 rank mask with output range {0,8}:
  
  ![pic1 ](https://res.cloudinary.com/dmfrqkuif/image/upload/v1592985597/Screenshot_from_2020-06-24_15-59-21_tkrckg.png)
  
   Notice the contrast has been improved significantly currently. In this repo, I set mask size = 11 and save values of r(x) into matrix: ranked_matrix. r(x) = number of elements with a lower value / number of elements examined.

4. (This part is different from Freddy' paper).

    In his paper, the number of divided segments is m. m = u + 1.2* v**(1/2). u is mean of gadient od D(n) and v is variance of D(n). D(n) means inside dencity of n segments. D(n) = sum of m number of sk / sum of m number of ak. Sk refers to the sum of rank values in one segment and a(start, end) = (end - start +1) ** 2. Optimal value of m occurs when gradient of D(n) decreases greatly.
    
    ![pic 2](https://res.cloudinary.com/dmfrqkuif/image/upload/v1592985597/Screenshot_from_2020-06-24_15-59-21_tkrckg.png)
    
    In his paper, optimal value of segments should be n. 
    
    But I found it boring to perform so much steps. Since the data of mine is short conversation, I decided to set n=3.
