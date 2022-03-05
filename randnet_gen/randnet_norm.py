n=int(input("Enter network size"))
alpha=int(input("Enter number of networks to be generated"))
import scipy.special

#directory='size{}'.format(n)
import numpy as np
import matplotlib.pyplot as plt

maxx=(n*(n+1))//2
maxalt=maxx-(n-1)  
edgearr=[0]*(maxx+1) 
tot=0
probarr=[0]*(maxx-(n-1) +1) 
for p in range(0,maxalt+1):
    pval=scipy.special.comb(maxalt,p)
    tot+=pval
    probarr[p]+=pval
   
probarr=np.array(probarr)
probarr=probarr/tot   
for p in range(1,maxalt+1):
    probarr[p]=probarr[p-1]+probarr[p]

for i in range(1+alpha,alpha+1+alpha):
    topofile=open("new/randomnet{}_{}.topo".format(n,i),"w")
    idsfile=open("new/randomnet{}_{}.ids".format(n,i),"w")
    topofile.write("Source Target Type\n")
    idsfile.write("node ids\n")
    link_matrix=np.zeros((n,n))
    cnt=0
    max1=(n*(n+1))//2
    
 
    q=0

   
    stcnt=1
    #generate spanning tree
    start= np.random.randint(0,n)
    st=[0]*n
    st[start]=1
    
    while(stcnt < n):   
     
       j=np.random.randint(0,n) 
       #print(stcnt,start,j)
       l=np.random.randint(1,3)
       if(st[j]!=1):
           st[j]=1
           stcnt+=1
           link_matrix[start][j]=l
       start=j
        
   
    
    edges=0
    p1=np.random.rand()
    for p in range(0,maxalt+1):
        if(p1<=probarr[p]):
            edges=p
            break
    q=0        
    while( q < edges ):  
        
        j=np.random.randint(0,n)
        k=np.random.randint(0,n)
        l=np.random.randint(1,3)
     
        if(link_matrix[j][k]==0):
            cnt=cnt+1
            q+=1
            link_matrix[j][k]=l
      
    cnt1=0   
    cnt+=n-1
    #print(cnt)
    try:
       edgearr[cnt]+=1
    except:
       print(cnt)
    
    for j in range(0,n):
        for k in range(0,n):
            if(link_matrix[j][k]!=0):
                cnt1=cnt1+1
                g1=(chr(j+65))
                g2=(chr(k+65))
                if(cnt1!=cnt):
                    topofile.write("{} {} {}\n".format(g1,g2,int(link_matrix[j][k])))
                else:
                    topofile.write("{} {} {}".format(g1,g2,int(link_matrix[j][k])))
          
    for j in range(0,n):
        g1=(chr(j+65))
        idsfile.write("{} {}\n".format(g1,j))
    idsfile.close()
    topofile.close()    
   
   
#xarr= [i for i in range(0,maxx+1)]
#plt.scatter(xarr,edgearr)
#plt.savefig("graph.jpg")    