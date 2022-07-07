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


def mod_link_matrix(n):
    global probarr
    global maxx
    global maxalt
    global tot
    link_matrix=np.zeros((n,n))
    cnt=0
    max1=(n*(n+1))//2    
    q=0
    stcnt=1
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
            q+=1
            link_matrix[j][k]=l   
    return link_matrix        
   


for p in range(1,maxalt+1):
    probarr[p]=probarr[p-1]+probarr[p]

for i in range(1,1+alpha):
    topofile=open("randomnet{}_{}_R3_fix.topo".format(n,i),"w")
    idsfile=open("randomnet{}_{}_R3_fix.ids".format(n,i),"w")
    topofile.write("Source Target Type\n")
    idsfile.write("node ids\n")

    flag = 0
    link_matrix = np.zeros((n,n))
    while(flag == 0):
        link_matrix = mod_link_matrix(n)
        flag3 = 0
        for j in range(n):
            flag2 = 0
            for k in range(n):
                if(link_matrix[j][k]!=0):
                    flag2 = 1
                    break
            if(flag2==0):
                flag3 = 1
                break
        if(flag3!=1):
            flag = 1
            
    cnt = 0   
    print(link_matrix)
    for j in range(0,n):
        for k in range(0,n):        
             if(link_matrix[j][k]!=0):
                cnt+=1
    cnt1 = 0
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
   
   
