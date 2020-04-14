import re 
from Preprocess.preProcess import *   
import string 



#input:Given a string in the form of "(start1-end1)span_text1||(start2-end2)......"  or "{}" if no span selected 
#output:Gives back a list of tuples of the form (string,start,end) or [] if no span selected

def giveSpanList(string1):
    if string1 in ["{}","{","}"]  :
        return []
    list1=string1.split("||")
    string_all=[]
    for l in list1:
    	# collect the string 
        string=re.sub(r'\([^)]*\)', '', l)
        # colect the string postion (start--end) in the original text
        string_mask=re.findall('\((.*?)\)',l)[0]
        [start,end]=string_mask.split("--")
        string_all.append((string,start,end))
    return string_all 

#INPUTS: text,mask_all,debug 
#text contains the text in the dataset
#mask all contains the list of attention span from all the annotaters
#debug is set to true if print results are required  
#OUTPUTS: outputs word tokens , 


def returnMask(text,mask_all,debug=False):
    ### convert mentions to @user
    text = re.sub('@\w+', '@user',text)
    
	#### remove any html spans if present in the text    	
    text=cleanhtml(text)

    ###remove some errorneous characters in the text
    text = re.sub(u"(\u2018|\u2019|\u201A|\u201B|\u201C|\u201D|\u201E)", "'", text)
    text = text.replace("\r\n",' ').replace("\n",' ')
   	
   	####initialize word mask and word tokens 
    word_mask_all=[]
    word_tokens_all=[]


    for mask in mask_all:

        spanlist=giveSpanList(mask)
        total_length=len(text)
        list_pos=[0]
        mask_pos=[0]
        
        spanlist.sort(key = lambda elem: int(elem[1]))
        
        for ele in spanlist:
            if(int(ele[1])!=0):
                list_pos.append(int(ele[1]))
            mask_pos.append(1)
            list_pos.append(int(ele[2]))
            mask_pos.append(0)
        if(list_pos[-1]!=total_length):
            list_pos.append(total_length)
        string_parts=[]
        for i in range(len(list_pos)-1):
            string_parts.append(text[list_pos[i]:list_pos[i+1]])
        
        
        word_tokens=[]
        word_mask=[]
        if(debug==True):     
            print(list_pos)
            print(string_parts)

        for i in range(0,len(string_parts)):
            tokens=ek_extra_preprocess(string_parts[i])
            masks=[mask_pos[i]]*len(tokens)
            word_tokens+=tokens
            word_mask+=masks
        word_mask_all.append(word_mask)
        word_tokens_all.append(word_tokens)
    #print(len(word_tokens_all))    
    return word_tokens_all[0],word_mask_all




if __name__== "__main__":
	print(returnMask(
			"Laura Loomer raped me while screaming at me in her disgusting kike language and said we must exterminate the goyim. #LauraLoomer #Loomergate",
			["(51--75)disgusting kike language||(93--114)exterminate the goyim","{}","{}"]))


