#include "common/common.h"
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <stdio.h>
#include<string.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <sstream>
#include<cuda.h>
#include<cuda_runtime.h>
#include<cstring>
	
using namespace std;

#define THREADS_PER_BLOCK  512
	
struct constraint
{
	string name;
	string op;
	int value;
	int filterid;
	int rowid;
};

struct newConstraint
{
	char  name[5];
	char  op[5];
	int value;
	int filterid;
	int rowid;
};
	
struct filter
{
	int size;
	unsigned int count;
	int filtid;
};
	
struct names
{
	string name;
	int rowid;
	int count;
};
	
struct input
{
	string name;
	string op;
	int value;
};

struct newInput
{
	char  name[5];
	char op[5];
	int value;
};

//Start of Cuda Kernel match

__global__ void match(newConstraint *constraints, int *sizec, filter *filters, newInput *inputs, int nlines)
{ 

	int x = blockIdx.x*blockDim.x+threadIdx.x;
 	int y = blockIdx.y;
	
	__shared__  char  name[5];
	__shared__ int inputValue;
	__shared__ int rowid;

	/*if(x==0&&y==0){
	printf("\nsize:%d",sizec[0]);
	printf("\nsize:%d",sizec[1]);
	printf("\nsize:%d",sizec[2]);
}*/
	if(threadIdx.x==0)
	{
		int i=0;
		while(inputs[y].name[i]!='\0'){
			name[i]=inputs[y].name[i];
			i++;
		}	
		name[i]='\0';
		
		inputValue=inputs[y].value;
		rowid=y;//
	}
	__syncthreads();
	
	//printf(" %d",r);
	if(x>=sizec[rowid])
	return;

	if(name[0]=='*')
	return;

	//printf("\n%d %d",blockIdx.y,threadIdx.x);
        //data of constraints
	
	int z=nlines*rowid+x;
	int valsub=constraints[z].value;
	
	//char constraintName[5]=constraints[nlines*rowid+x].name;
	
	char op1[5];
	int i=0;
	while(constraints[z].op[i]!='\0'){
			op1[i]=constraints[z].op[i];
			i++;
		}	
	op1[i]='\0';
	
	int filterid=constraints[z].filterid;
//data of shared input
	int valpub=inputValue;
	
	/*if(threadIdx.x==0&&blockIdx.y==0)
	printf("\n%d %s %d %s",inputValue,name,valpub,op1);
	*/
		int satisfied = 0;
	
			if(op1[0]=='<' && op1[1]=='=')
			{
				if(valpub <= valsub)
					satisfied =1;
			}
			else if(op1[0]=='<')
			{
				if(valpub < valsub)
					satisfied =1;
						
			} 
			else if(op1[0]=='>' && op1[1]=='=')
			{
				if(valpub >= valsub)
					satisfied =1;
			}
			else if(op1[0]=='>')
			{

				if(valpub > valsub)
					satisfied =1;
					
			}
			else if(op1[0]=='=')
			{
					
				if(valpub == valsub)
					satisfied =1;
			}
			else if(op1[0]=='!' && op1[1]=='=')
			{
				if(valpub != valsub)
					satisfied =1;
					
			}

			if(satisfied==0)
			return;
                
			atomicInc(&filters[filterid].count, filters[filterid].size+1);
			//filters[filterid].count++;
			//__syncthreads();
			//filters[filterid].filtid = filterid+1;
			//int q = 0;
			//for( ; q < filters.size() ; q++)
			//if(filters[filterid].count == filters[filterid].size)
			        
			
}

//End of kernel

/*char CONSTRAINT_FILE[] = "subs_20000.txt";
char EVENTS_FILE [] = "pubs_1000.txt";*/

char CONSTRAINT_FILE[] = "b.txt";
//CONSTRAINT_FILE =new char["1st_1_lack.txt";
char EVENTS_FILE [] = "a.txt";
	
int contains(string line, vector<names> namesArray)
{
	int length = namesArray.size();
	
	for(int i=0;i<length;i++)
	{
		names name = namesArray[i];
		 
		if(name.name==line)
		{
			return i;
		}
	}
	return -1;
}
	
void printStrings(vector<string> listt)
{
	int length = listt.size();
	
	for(int i=0;i<length;i++)
	{
		cout<<"\n"<<listt[i];
	}
	return;
}
	
	//Functions to split filter by ';'
std::vector<std::string> &split(const std::string &s, char delim,std::vector<std::string> &elems) {
	std::stringstream ss(s);
	std::string item;
	while (std::getline(ss, item, delim)) {
		elems.push_back(item);
	}
	return elems;
}
	
std::vector<std::string> split(const std::string &s, char delim) 
{
	std::vector<std::string> elems;
	split(s, delim, elems);
	return elems;
}

	
vector<filter> getFilters(vector <string> c_list , vector<constraint> constraints)
{
	vector<filter> filters;
	int length = c_list.size();
	filter tempFilter;
	for(int i=0;i<length;i++)
	{
		vector <string> splitted = split(c_list[i],';');
		tempFilter.size=splitted.size();		
		tempFilter.count=0;
		tempFilter.filtid= i ;
		filters.push_back(tempFilter);
	}
	return filters;
}
	
	//End of split function
	
	/*vector<string> extractNames(vector<string> c_list){
	vector <string> names;
	for(int i=0;i<c_list.size();i++){
	string filter = c_list[i];
	
	vector <string> constraints = split(filter,';');
	
	for(int j=0;j<constraints.size();j++){
	string constraint = constraints[j];
	if(!contains(constraint,names))
	names.push_back();
	}
	
	}
	}*/
int getIntFromString(string value)
{
	int length = value.size();
	
	int val=0;
	for(int i=0;i<length;i++)
	{
		int charval = value[i]-48;
		val =val *10 +charval;
	}
	return val;
}
	
void addName(string name,vector<names> &namesArray)
{
	if(contains(name,namesArray))
	return;
	names newName;
	newName.name=name;
	newName.count=1;
	newName.rowid=namesArray.size();
}

	
	
vector<constraint> getConstraints(vector <string> c_list)
{
	int length = c_list.size();
	
	vector<constraint> constraints;
	
	for(int i=0;i<length;i++)
	{
	constraint tempConstraint;
	vector<string> splitted = split(c_list[i],';');
	int splitLength = splitted.size();
		for(int j=0;j<splitLength;j++)
		{
			string eachConstraint= splitted[j];
			string name,op,value;
			int filterid=i;
	
			int constraintLength = eachConstraint.size();
	
			for(int k=0;k<constraintLength;k++)
			{
				char temp = eachConstraint[k];
				if((temp>=65&&temp<=90)||(temp>=97&&temp<=122))
				name+=temp;
				else if(temp>=48&&temp<=57)
					value+=temp;
				else
					op+=temp;
	
			}
	
			int attrValue = getIntFromString(value);
	
			tempConstraint.name=name;
			tempConstraint.op=op;
			tempConstraint.filterid=filterid;
			tempConstraint.value=attrValue;
			//tempConstraint.rowid=filterid; //modified
			constraints.push_back(tempConstraint);
		}
	}
	return constraints;
	
	
}
	
void getSizeC(vector<names> namesArray,int * const sizeC, int * maxVal)
{
	int length = namesArray.size();
	
	for(int i=0;i<length;i++)
	{
		sizeC[i]=namesArray[i].count;
		if(sizeC[i]>*maxVal)
		*maxVal=sizeC[i];
	}
	return;
	
}
	
void printFilters(vector<filter> filters)
{
	int length = filters.size();
	
	for(int i=0;i<length;i++)
	{
		printf("\nFilterSize:%d FilterCount:%d FiltRowid:%d",filters[i].size,filters[i].count,filters[i].filtid);
	}
	return;
}

void printConstraints(vector<constraint> constraints)
{
	int length = constraints.size();
	
	
	for(int i=0;i<length;i++)
	{
	
	cout<<"\n"<<i<<" : "<<"name "<<constraints[i].name<<" op "<<constraints[i].op<<" filterid "<<constraints[i].filterid;
	cout<<" value "<<constraints[i].value;//<<" rowid "<<constraints[i].rowid;
	}
	return;
}
	

	
void printNames(vector<names> namesArray)
{
	int length = namesArray.size();
	for(int i=0;i<length;i++)
	{
		cout<<"\n"<<namesArray[i].name<<" "<<namesArray[i].rowid<<" "<<namesArray[i].count;
	}
	//cout<<"LLLL: "<<length<<endl;
}
	
vector<names> getNames(vector<constraint> &constraints)
{
	vector <names> namesArray;
	int length = constraints.size();
	
	for(int i=0;i<length;i++)
	{
		string attrName = constraints[i].name;
		
		int id=contains(attrName,namesArray);
		if(id!=-1)
		{
			namesArray[id].count++;
			constraints[i].rowid=id;
		}
		else
		{
			names tempName;
			tempName.name=attrName;
			tempName.count=1;
			tempName.rowid=namesArray.size();
			constraints[i].rowid=namesArray.size();
	
			namesArray.push_back(tempName);
	
		}
	}
	return namesArray;
}

void printInput(vector<input> inputEvents)
{
	for(int i=0;i<inputEvents.size();i++)
	{
		cout<<"\n name "<<inputEvents[i].name<<" op "<<inputEvents[i].op<<" value "<<inputEvents[i].value;
	}
}
	
void setRowIds(vector <constraint> &constraints,vector<names> names)
{
	for(int i=0;i<constraints.size();i++)
	{
		int match=0;
		for(int j=0;j<names.size();j++)
		{
			if(names[j].name==constraints[i].name)
			{
				constraints[i].rowid=j;
				match=1;
				break;
			}
		}
		if(match==0)
		constraints[i].rowid=101;
	}	
}
	
int main()
{
	vector<string> c_list , e_list;
	float resultTime = 0.0f;
	//Events and Constraints reading
	ifstream in_stream;
	string line;

	in_stream.open(EVENTS_FILE);
	while(!in_stream.eof())
	{
		in_stream >> line;
		e_list.push_back(line);
	}
	in_stream.close();
	
	in_stream.open(CONSTRAINT_FILE);
	while(!in_stream.eof())
	{
		in_stream >> line;
		c_list.push_back(line);
	}
	in_stream.close();

	c_list.erase(c_list.begin()+c_list.size()-1);
	//cout<<*(c_list.begin())<<endl;
	e_list.erase(e_list.begin()+e_list.size()-1); 
	//printStrings(c_list);
	//printStrings(e_list);
	int ii=0,cSize=c_list.size(),index=0;
	
	//int results[cSize];
	//for(int k=0;k<cSize;k++)results[k]=0;
	int MAX_LINES=200000;
        cout<<"Filters Size : "<<cSize <<endl;
	cout<<"Events Size : "<<e_list.size()<<endl;
       	while(true)
	{
	        if(ii>=cSize)
			break;
		vector <string> cc_list;
		for(;(ii)<((index+1)*MAX_LINES)&&((ii)<cSize);ii++)
		{
			cc_list.push_back(c_list[ii]);
		
		}	
                vector<constraint> constraints = getConstraints(cc_list);
	        vector<filter> filters = getFilters(cc_list,constraints);         
	        vector <names> namesArray = getNames(constraints);
	//cout<<"\n\n"<<"Filters";
	//printFilters(filters);
	//cout<<"\n\n"<<"Constraints";
	//printConstraints(constraints);
	
	/*cout<<"\n\n"<<"Names Array";
	printNames(namesArray);//name , rowid , count 100;*/
	/*cout<<"\n\n"<<"SizeC";
	for(int i=0;i<namesArray.size();i++)
	cout<<"\n"<<sizeC[i];*/
	//Start of cuda code
	
	//cout<<"GridSize"<<gridSize.y;
	//cout<<"MaxC: "<<namesArray.size();
	cudaEvent_t start_event, stop_event;
	float time=0.0f, timefinal=0.0f;
	
	int ngpus;
	
	cudaGetDeviceCount(&ngpus);	
	cout<<"CUDA-capable devices : "<<ngpus<<endl;
	
	/*int **sizeC = (int **)malloc(sizeof(int *)* ngpus) ;
	int *maxConstraints = new int [ngpus];

	for(int device = 0 ; device < ngpus ; device++)
        {
                sizeC[device] =(int*)malloc(sizeof(int)*namesArray.size());
                getSizeC(namesArray,sizeC[device],&maxConstraints[device]);
	        
        }       	        
	int maxC = 2145 ;*/
	
	int *sizeC , maxConstraints=0;
	sizeC = new int[namesArray.size()];
	getSizeC(namesArray,sizeC,&maxConstraints);
	/*cout<<"\n\n"<<"SizeC";
	for(int i=0;i<namesArray.size();i++)
	{
	        cout<<namesArray.at(i);
	        cout<<":"<<sizeC[i]<<endl;
	}*/
	
	/*for(int i = 0 ; i<ngpus ; i++)
        {
                cout<<"maxCons : "<<maxConstraints[i]<<endl;
                if(maxC < maxConstraints[i] )
                        maxC = maxConstraints[i];
        }*/
        //cout<<"maxC : "<<maxConstraints<<endl;  2145
        
	dim3 dimBlock;
	dimBlock.x=512;
	dimBlock.y=1;
	
	
	dim3 gridSize;
	gridSize.x = (maxConstraints/dimBlock.x)+1;

	gridSize.y = namesArray.size();
		
	int separatedConstraintsSize = maxConstraints*namesArray.size();
	vector <constraint> separatedConstraints;
	separatedConstraints.reserve(separatedConstraintsSize);
	
	int constraintsCounters[namesArray.size()];
	
	for(int i=0;i<namesArray.size();i++)
		constraintsCounters[i]=0;
		
	constraint dummyConstraint;
	dummyConstraint.name="*";
	dummyConstraint.filterid=1;
	dummyConstraint.op="=";
	dummyConstraint.value=100;
	dummyConstraint.rowid = 0;

	for(int i=0;i<separatedConstraintsSize;i++)
		separatedConstraints.push_back(dummyConstraint);
	//logic to form namevector
	for(int i=0;i<constraints.size();i++)
	{
		constraint tempConstarint=constraints[i];
		separatedConstraints[tempConstarint.rowid*maxConstraints+constraintsCounters[tempConstarint.rowid]]=tempConstarint;
		constraintsCounters[tempConstarint.rowid]++;
	}
	//Done with constraints
	/*cout<<"\n\n"<<"Sepatrated Constarints";
	printConstraints(separatedConstraints);*/        
       	//Copy Constraints to struct array
	//newConstraint *allConstraints; //[separatedConstraintsSize];
        newConstraint **allConstraints = (newConstraint **)malloc(sizeof(newConstraint *) * ngpus);
        //allConstraints = new newConstraint[separatedConstraintsSize];
        int p = 0;        
        for(int device = 0 ; device < ngpus ; device++)
                allConstraints[device] = new newConstraint[(separatedConstraintsSize)/ngpus];
                  
                
        for(int device = 0 ; device < ngpus ; device++)
        {
                if(p < separatedConstraintsSize/ngpus)
                        for( ; p < separatedConstraintsSize/ngpus ; p++)
	                {
	        		int size=separatedConstraints[p].name.length();		
	        		int j=0;
	        		for(;j<size;j++){
	        		allConstraints[device][p].name[j]=separatedConstraints[p].name[j];	
			        }
			        allConstraints[device][p].name[j]='\0';
			        j=0;
			        int size1=separatedConstraints[p].op.length();

			        for(;j<size1;j++){
			        allConstraints[device][p].op[j]=separatedConstraints[p].op[j];	
			        }
			        allConstraints[device][p].op[j]='\0';

		                allConstraints[device][p].value=separatedConstraints[p].value;
		                allConstraints[device][p].filterid=separatedConstraints[p].filterid;
		                allConstraints[device][p].rowid=separatedConstraints[p].rowid;
	                }
	          else
	          {
	                int i = 0;
	                //cout<<p<<endl;exit(1);
	                for( ; p < separatedConstraintsSize; p++ )
	                {
	                                            
	                        int size=separatedConstraints[p].name.length();		
	        		int j=0;
	        		for(;j<size;j++){
	        		allConstraints[device][i].name[j]=separatedConstraints[p].name[j];	
			        }
			        //cout<<p<<" : "<<allConstraints[device][i].name<<"";
			       
			        allConstraints[device][i].name[j]='\0';
			        j=0;
			        int size1=separatedConstraints[p].op.length();

			        for(;j<size1;j++){
			        allConstraints[device][i].op[j]=separatedConstraints[p].op[j];	
			        }
			        allConstraints[device][i].op[j]='\0';
                                        
		                allConstraints[device][i].value=separatedConstraints[p].value;
		                
		                allConstraints[device][i].filterid=separatedConstraints[p].filterid;
		                
		                allConstraints[device][i].rowid=separatedConstraints[p].rowid;
		                
		                i++;
	                }	              
	          }               
        }
        
        /*for(int i = 0 ; i < separatedConstraintsSize/ngpus ; i++)
        {
                      cout<<i<<">"<<allConstraints[1][i].name<<" "<<allConstraints[1][i].filterid<<endl;
                      
        }          
        exit(1);*/
	/*int x =0;
	cout<<"\n\n"<<"Names of separated cs ";
	for(int device = 0 ; device < ngpus ; device++)
	{
	        if(x < separatedConstraintsSize/ngpus-1)
	       { 
	                for( ; x < separatedConstraintsSize/ngpus ; x++)
	                {
		                cout<<x<<" : "<<allConstraints[device][x].name<<" ";
	                }       
	                cout<<endl<<endl;
	       }
	        else
	        //while(x>separatedConstraintsSize/ngpus)
	        {
	                for( ;  x < separatedConstraintsSize ; x++)
		        cout<<x<<" : "<<allConstraints[device][x].name<<endl;
	                x++;
	        }
	        
	}
	exit(1);*/
	//End of constraint copy to array
	//newConstraint **d_allConstraints;
	newConstraint **d_allConstraints = (newConstraint **)malloc(sizeof(newConstraint *) * ngpus);
	
	int **d_sizeC =(int**)malloc(sizeof(int *) * ngpus) ;
	//Stream for asynchronous command execution
           cudaStream_t *stream = (cudaStream_t *)malloc(sizeof(cudaStream_t) * ngpus);//non-default stream is declared.
        //copy filters to struct array
	//filter *allFilters;
	//allFilters = new filter[filters.size()*ngpus];
	
	/*filter **resultFilt =(filter **)malloc(sizeof(filter *)*ngpus);
	for(int ndev = 0 ; ndev < ngpus ; ndev++)
	{
	        resultFilt[ndev] = new filter[filters.size()/ngpus];
	}*/
	//filter *resultFilt;
	//resultFilt = new filter[sizeof(filter) * filters.size()]; 
	
	
	
	/*cout<<"\n\n"<<"Filters Size";
	for(int device = 0; device < ngpus ; device++ )
	for(int i=0;i<filters.size();i++)
	{
		cout<<i<<":"<<allFilters[device][i].count<<" "<<allFilters[device][i].size<<" "<<allFilters[device][i].filtid<<endl;;
	}
	exit(1);*/
	
	//End of copy filters to struct array
	
        filter **d_allFilters = (filter **)malloc(sizeof(filter *) * ngpus);// * filters.size());
        
          for(int device = 0 ; device < ngpus ;device++)
                {
                        cudaSetDevice(device);
                        cudaMalloc((void **) &d_allConstraints[device], (separatedConstraintsSize*sizeof(struct newConstraint))/ngpus);
                        (cudaMalloc((void **) &d_sizeC[device],namesArray.size()*sizeof(int)));
                        //CHECK(cudaMalloc((void **)&resultFilt[device],filters.size()/ngpus));
                         //CHECK(cudaMemSet(resultFilt[device], 0 , sizeof(filter)));
                        (cudaStreamCreate(&stream[device]));
                         
                        // CHECK(cudaMallocHost((void **)&allFilters[device],filters.size()));
                
                }
                
                for(int ndevice = 0 ; ndevice < ngpus ; ndevice++)
                {
                  cudaSetDevice(ndevice);
                 	(cudaMemcpyAsync(d_allConstraints[ndevice],allConstraints[ndevice],separatedConstraintsSize/ngpus*sizeof(struct newConstraint), cudaMemcpyHostToDevice, stream[ndevice]));    

               	       	(cudaMemcpyAsync(d_sizeC[ndevice],sizeC,namesArray.size()*sizeof(int),cudaMemcpyHostToDevice,stream[ndevice]));
                }

        	//Input part starts
        	input dummyInput;
        	dummyInput.name="*";
        	dummyInput.op="=";
        	dummyInput.value=100;
	
        	int *finalres;
        	finalres = new int[filters.size()]; //for saving results
	        
	        //for(int i=0;i<e_list.size();i++)
	        for(int i = 0 ; i <3  ; i++)
	        {
	        	vector <string> inputString;
	        	inputString.push_back(e_list[i]);
		
	        	vector <constraint> inputConstraints = getConstraints(inputString);//event separating
	        	
	        	//printConstraints(inputConstraints);
	        	//exit(1);
	        	setRowIds(inputConstraints,namesArray);
	        
	        	vector <input> inputEvents;
	        	for(int j=0;j<namesArray.size();j++)
	        	{
	        		inputEvents.push_back(dummyInput);
	        	}
	             for(int j=0;j<inputConstraints.size();j++)
	        	{
	        		input tempInput;
	        		tempInput.name=inputConstraints[j].name;
	        		tempInput.op=inputConstraints[j].op;
	        		tempInput.value=inputConstraints[j].value;
	        		inputEvents[inputConstraints[j].rowid]=tempInput;
	        	}
	        
	        	//printInput(inputEvents);
	        	//exit(1);
	        	newInput *allInputs ;//[inputEvents.size()] ;
                        allInputs = new newInput[inputEvents.size()];
	                //newInput **allInputs = (newInput **)malloc(sizeof(newInput *)*ngpus ) ; // * inputEvents.size());
        		
        		    //    allInputs[ndevice] = new newInput[inputEvents.size()];
        		        for(int jj=0;jj<inputEvents.size();jj++)
        		        {
			
        			        int size=inputEvents[jj].name.length();
        			        int j=0;
        			        for(;j<size;j++){
        			        allInputs[jj].name[j]=inputEvents[jj].name[j];	
       			                }
			                allInputs[jj].name[j]='\0';
			                j=0;
			                int size1=inputEvents[jj].op.length();

			                for(;j<size1;j++){
			                allInputs[jj].op[j]=inputEvents[jj].op[j];	
			                }
			                allInputs[jj].op[j]='\0';

        			        allInputs[jj].value=inputEvents[jj].value;			
		                } 		
	   
		/*cout<<"\n\n"<<"InputEvents";
		static int xxx = 0;
		//for(int d = 0 ;d < ngpus ; d++)
		for(int i=0;i<inputEvents.size();i++)
		{
			cout<<i<<":"<<allInputs[i].name<<" "<<allInputs[i].op<<" "<<allInputs[i].value<<endl ;
			xxx++;
		}
		cout<<xxx<<endl;*/
		//exit(1);
	        //continue;
		filter **allFilters = (filter **)malloc(sizeof(filter *) * ngpus);

                for(int device = 0 ; device < ngpus ; device++)
                {
                        allFilters[device] = new filter[filters.size()];                
                        for(int i=0 ; i < filters.size() ; i++)
	                {
		                allFilters[device][i].size=filters[i].size;
		                allFilters[device][i].count=filters[i].count;
		                allFilters[device][i].filtid=0;
	                }
                }       
				
                newInput **d_allInputs = (newInput **)malloc(sizeof(newInput *) * ngpus);
                
                for(int ndevice = 0 ; ndevice < ngpus ;ndevice++)
                {
                        CHECK(cudaSetDevice(ndevice));
                        cudaDeviceProp devProp;
                        cudaGetDeviceProperties(&devProp , ndevice);
                        cout<<"Device "<<devProp.name<< " has compute capability : "<< devProp.major<<"."<<devProp.minor<<endl;

	       		(cudaMalloc((void **)&d_allFilters[ndevice],(filters.size()*sizeof(struct filter))));	       
	                (cudaMemcpy(d_allFilters[ndevice], allFilters[ndevice], filters.size()*sizeof(struct filter), cudaMemcpyHostToDevice));
                          
                        (cudaMalloc((void **)&d_allInputs[ndevice], inputEvents.size()*sizeof(struct newInput)));		
	 	    
		        (cudaMemcpyAsync(d_allInputs[ndevice], allInputs, inputEvents.size()*sizeof(struct newInput), cudaMemcpyHostToDevice,stream[ndevice]));
	                 
	                cudaEventCreate(&start_event) ;
		        cudaEventCreate(&stop_event) ;
		        cudaEventRecord(start_event, stream[ndevice]);
   		        match<<<gridSize,dimBlock,0,stream[ndevice]>>>(d_allConstraints[ndevice], d_sizeC[ndevice], d_allFilters[ndevice], d_allInputs[ndevice], maxConstraints);
   		        //match<<<(separatedConstraintsSize/ngpus)/THREADS_PER_BLOCK,THREADS_PER_BLOCK,0,stream[ndevice]>>>(d_allConstraints[ndevice], d_sizeC[ndevice], d_allFilters[ndevice], d_allInputs[ndevice], maxConstraints);
	                
		        (cudaThreadSynchronize());
		        
                        //(cudaStreamQuery(stream[ndevice]));
		        //timefinal+=time;
		        cudaEventRecord(stop_event, stream[ndevice]);
		        cudaEventSynchronize( stop_event);
		        cudaEventElapsedTime( &time, start_event, stop_event );
		        cudaEventDestroy( start_event ); // cleanup
		        cudaEventDestroy( stop_event );
                        timefinal+=time;
		        //printf("\ndone and it took: %f milliseconds\n", time);
		        //we want count of filter
    		        (cudaMemcpy(allFilters[ndevice], d_allFilters[ndevice], filters.size()*sizeof(struct filter), cudaMemcpyDeviceToHost)); 		       
    		       
    		         //CHECK(cudaStreamSynchronize(stream(ndevice)));
    		       
    		        (cudaStreamQuery(stream[ndevice]));
    		        //if(allFilters[ndevice][i].count == allFilters[ndevice][i].size)
    		         //cout<<e_list.at(i)<<" : "<<allFilters[ndevice][i].count<<endl;  
    		         //cout<<e_list.at(i)<<" : "<<allFilters[ndevice][i].count<<endl;  
    		              		            		
    		        cout<<"Time Required : "<<time<<endl;
	           }      
	          
	          //for(int device = 0 ; device < ngpus ; device++)
	           /*{
	                for(int p = 0; p <filters.size() ; p++)
	                        if(allFilters[0][p].count > 0 || allFilters[1][p].count > 0 )
	                        cout<<p<<":"<<allFilters[0][p].count<<" "<<allFilters[1][p].count<<"\n";
	                cout<<endl;        
	           }*/
	          cout<<endl<<e_list.at(i)<<endl;
	         // exit(1); 
    	        /*int count= 0 ;
    		//for(int ndevice = 0 ; ndevice < ngpus; ndevice++)
    		for(int x = 0 ; x < filters.size() ; x++)
	        {
	                int counter = 0 ;
	                counter = allFilters[0][x].count + allFilters[1][x].count ;    
	                if(counter == allFilters[0][x].size)
	                {
	                         cout<<x+1 <<" : "<<e_list.at(i)<<" count : "<<counter<<endl;  
	                         count++;
	                }
	               
	        }
	        cout<<"count : "<<count<<endl;*/
	
	                 //  int counter = 0 ;
	                //  counter = allFilters[0][i].count + allFilters[1][i].count;      
    		         //   allFilters[0][i].count =counter;
    		          //  allFilters[1][i].count  =counter;    
    		                 
	        
                /*for(int dev = 0 ; dev < ngpus ;dev++)
                 {
                        for(int y = 0 ; y < filters.size() ; y++)
                        if(allFilters[dev][y].count == allFilters[dev][y].size)  
    		            		   cout<<e_list.at(i)<<" : "<<allFilters[dev][y].count<<endl;
    		         cout<<endl;   		   
                 }*/                
                
        
                for(int ndevice = 0 ;ndevice < ngpus ; ndevice++)
		{
		        cudaSetDevice(ndevice);
		        cudaStreamSynchronize(stream[ndevice]);
		}
                  		
                
                filter *allFilt;
                allFilt = new filter[filters.size()];
                int p;
                for(p = 0; p < filters.size() ; p++)
                {
                        
                        allFilt[p].count = allFilters[0][p].count + allFilters[1][p].count; 
                } 
                
               //cout<<p<<": "<<allFilt[6011].count<<endl;
                /*for(int p = 0 ; p < 6013; p++)
                        cout<<p<<":"<<allFilters[0][p].count<<" "<<allFilters[1][p].count<<" "<<allFilt[p].count<<endl;
                cout<<endl<<endl;*/
               //exit(1);
                int *results;
                results = new int[cSize];
                
                //for(int ndevice = 0 ; ndevice < ngpus ; ndevice++)
                {
                                 for(int q = 0; q < filters.size();q++)
		                {
                			//cout<<allFilters[ndevice][i].count<<endl;
	                		//continue;
			                int res=allFilt[q].count/allFilters[0][q].size;
			                finalres[q]+=res;
			                results[q]+=res;
			                allFilt[q].count=0;
			                //printf("%d \n ",finalres[i]);
		                }                     
                }
                	
                	
                int count1 = 0;
        	for(int i=0;i<cSize;i++)
	        {
	        	if(results[i] >0)
	        	{
	        	        printf("\nResult-> %d %d ",i,results[i]);
	                        count1++;
	                }
	        }	
		for(int ndevice = 0 ;ndevice < ngpus ; ndevice++)
		{
		        cudaSetDevice(ndevice);
		        cudaFree(d_allInputs[ndevice]);
                	cudaFree(d_allFilters[ndevice]);
		        
		}
		
	        delete[] allInputs ;
	        for(int dev =0 ; dev < ngpus ; dev++)
        	        delete[] allFilters[dev];
        	delete[] allFilters;
        	delete[] allFilt;
        	delete[] results;       
        	cout<<"count is :"<<count1<<endl;
	        //exit(1);
	       /* for(int dev  = 0 ; dev < ngpus ; dev++)
	        {
	                 for(int x = 0 ; x < filters.size() ; x++)
	                {
    		          if(allFilters[dev][x].size == allFilters[dev][x].count)
    		          {
    		                     cout<<e_list.at(i);   
    		                     cout<<" : "<<allFilters[dev][x].filtid<<": "<<allFilters[dev][x].count<<endl; 
    		          } 
	                 }
	                cout<<endl<<endl;
	        }*/    	
              
	}
	
	//End of loop for input
	index++;
	//printf("\ndone and it took: %f milliseconds\n", timefinal);
	resultTime+=timefinal;
	        cout<<"\nresultTime : "<<resultTime<<endl;
	         //int count = 0;
        	/*for(int i=0;i<cSize;i++)
	        
	        {
	        	if(results[i] >0)
	        	{
	        	        printf("\nResult-> %d %d ",i,results[i]);
	                        count++;
	                }
	        }*/
	        //cout<<"\n Count = "<<count<<endl;
	        for(int device  = 0 ; device < ngpus ; device++)
	        {       
	                cudaSetDevice(device);
	                cudaFree(d_allConstraints[device]);
	                cudaFree(d_sizeC[device]); 
	        }    
	        exit(1);
	}
		
	//printf("\ndone and it took: %f milliseconds\n",resultTime);
        /*int count = 0;
	for(int i=0;i<cSize;i++)
	
	{
		if(results[i] >0)
		{
		        printf("\nResult-> %d %d ",i,results[i]);
	                count++;
	        }
	}
	cout<<"\n Count = "<<count<<endl;
        cout<<" done and it took:"<< resultTime<<" milliseconds \n";
	*/
	return 0;
}
