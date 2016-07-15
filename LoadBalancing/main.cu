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
	
using namespace std;
	
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

__global__ void match(newConstraint *constraints,int *sizec,filter *filters,newInput *inputs,int nlines)
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
		rowid=y;
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

			atomicInc(&filters[filterid].count,filters[filterid].size);
			
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

	
vector<filter> getFilters(vector <string> c_list)
{
	vector<filter> filters;
	int length = c_list.size();
	filter tempFilter;
	for(int i=0;i<length;i++)
	{
		vector <string> splitted = split(c_list[i],';');
		tempFilter.size=splitted.size();
		tempFilter.count=0;
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
	constraint tempConstraint;
	vector<constraint> constraints;
	
	for(int i=0;i<length;i++)
	{
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
			tempConstraint.rowid=i; //modified
			constraints.push_back(tempConstraint);
		}
	}
	return constraints;
	
	
}
	
void getSizeC(vector<names> namesArray, int * sizeC, int * maxVal)
{
	int length = namesArray.size();
	
	for(int i = 0 ; i < length ; i++)
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
		printf("\n%d : FilterSize:%d FilterCount:%d",i, filters[i].size,filters[i].count);
	}
	return;
}

void printConstraints(vector<constraint> constraints)
{
	int length = constraints.size();
	
	
	for(int i=0;i<length;i++)
	{
	
	cout<<"\n"<<i<<" : "<<"name "<<constraints[i].name<<" op "<<constraints[i].op<<" filterid "<<constraints[i].filterid;
	cout<<" value "<<constraints[i].value<<" rowid "<<constraints[i].rowid;
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
	for(int i=0;i<5;i++)
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
	vector<string> c_list,e_list;
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
	//exit(1);
	e_list.erase(e_list.begin()+e_list.size()-1);
       
	//printStrings(c_list);
	//printStrings(e_list);
	int ii=0,cSize=c_list.size(),index=0;
	
	int results[cSize];
	for(int k=0;k<cSize;k++)results[k]=0;
	int MAX_LINES=200000;

	cout<<"Constraints Size : "<<cSize <<endl;
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

	vector<filter> filters = getFilters(cc_list);
	vector<constraint> constraints = getConstraints(cc_list);
	vector <names> namesArray = getNames(constraints);
	//printNames(namesArray);
	/*cout<<"\n\n"<<"Filters";
	printFilters(filters);
	cout<<"\n\n"<<"Constraints";
	printConstraints(constraints);
	cout<<"\n\n"<<"Names Array";*/
	//printNames(namesArray);//name , rowid , count 100
	
	int sizeC[namesArray.size()], maxConstraints=0;
	getSizeC(namesArray,sizeC,&maxConstraints);
	
	/*cout<<"\n\n"<<"SizeC";
	for(int i=0;i<namesArray.size();i++)
	cout<<"\n"<<sizeC[i];*/ 
	
	//cout<<"maxC : "<<maxConstraints<<endl;
	
	int separatedConstraintsSize = maxConstraints*namesArray.size();
	vector <constraint> separatedConstraints;
	separatedConstraints.reserve(separatedConstraintsSize);
	
	int constraintsCounters[namesArray.size()];
	
	for(int i=0;i<namesArray.size();i++)
	{
		constraintsCounters[i]=0;
	}
	
	constraint dummyConstraint;
	dummyConstraint.name="*";
	dummyConstraint.filterid=1;
	dummyConstraint.op="=";
	dummyConstraint.value=100;

	
	for(int i=0;i<separatedConstraintsSize;i++)
	{
		separatedConstraints.push_back(dummyConstraint);
	}
	
	
	
	for(int i = 0 ; i < constraints.size() ; i++)
	{
		constraint tempConstarint=constraints[i];
		separatedConstraints[tempConstarint.rowid*maxConstraints+constraintsCounters[tempConstarint.rowid]]=tempConstarint;
		constraintsCounters[tempConstarint.rowid]++;
	}
	
	//Done with constraints

	//cout<<"\n\n"<<"Sepatrated Constarints";
	//printConstraints(separatedConstraints);
        
	//Copy Constraints to struct array
	newConstraint allConstraints[separatedConstraintsSize];

	for(int i=0;i<separatedConstraintsSize;i++)
	{
			int size=separatedConstraints[i].name.length();		
			int j=0;
			for(;j<size;j++){
			allConstraints[i].name[j]=separatedConstraints[i].name[j];	
			}
			allConstraints[i].name[j]='\0';
			j=0;
			int size1=separatedConstraints[i].op.length();

			for(;j<size1;j++){
			allConstraints[i].op[j]=separatedConstraints[i].op[j];	
			}
			allConstraints[i].op[j]='\0';

		allConstraints[i].value=separatedConstraints[i].value;
		allConstraints[i].filterid=separatedConstraints[i].filterid;
		allConstraints[i].rowid=separatedConstraints[i].rowid;
	}
	
	//cout<<"\n\n"<<"Names of separated cs ";
	/*for(int i=0;i < 100;i++)
	{
		cout<<"\n"<<allConstraints[i].name;
	}
	exit(1);*/
	//End of constraint copy to array

	//copy filters to struct array

	filter allFilters[filters.size()];
	
	for(int i=0;i<filters.size();i++)
	{
		allFilters[i].size=filters[i].size;
		allFilters[i].count=filters[i].count;
	}

	/*cout<<"\n\n"<<"Filters Size";
	for(int i=0;i<filters.size();i++)
	{
		cout<<"\n"<<allFilters[i].size;
	}*/

	//End of copy filters to struct array

	//Start of cuda code

	dim3 dimBlock;
	dimBlock.x=512;
	dimBlock.y=1;
	
	dim3 gridSize;
	gridSize.x = (maxConstraints/dimBlock.x)+1;

	gridSize.y = namesArray.size();
	
	//cout<<"GridSize"<<gridSize.y;
	//cout<<"MaxC: "<<namesArray.size();
	cudaEvent_t start_event, stop_event;
	float time=0.0f,timefinal=0.0f;
	
	
	
	newConstraint *d_allConstraints;
	int *d_sizeC;
	
	
	cudaSetDevice(0);
	
	cudaDeviceProp devProp;
                cudaGetDeviceProperties(&devProp , 0);
                cout<<"Device "<< " has compute capability : "<< devProp.major<<"."<<devProp.minor<<endl;
	
	
	cudaMalloc((void **)&d_allConstraints,separatedConstraintsSize*sizeof(struct newConstraint));
	if(cudaSuccess!=cudaGetLastError())
        {
           printf("\nerror at cudamalloc11:%s",cudaGetErrorString(cudaGetLastError()));
        }

	cudaMemcpy(d_allConstraints,allConstraints,separatedConstraintsSize*sizeof(struct newConstraint),cudaMemcpyHostToDevice);

	cudaMalloc((void **)&d_sizeC,namesArray.size()*sizeof(int));
	cudaMemcpy(d_sizeC,sizeC,namesArray.size()*sizeof(int),cudaMemcpyHostToDevice);

	
	
	//End of cuda code

	

	//Input part starts
	input dummyInput;
	dummyInput.name="*";
	dummyInput.op="=";
	dummyInput.value=100;
	
	int finalres[filters.size()];//for saving results
	for(int i=0;i<filters.size();i++)
	{
		finalres[i]=0;
	}
	
	for(int i=6;i<10;i++)
	{
		vector <string> inputString;
		inputString.push_back(e_list[i]);
		
		vector <constraint> inputConstraints = getConstraints(inputString);//event separating
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
		newInput allInputs[inputEvents.size()];
	
		for(int i=0;i<inputEvents.size();i++)
		{
			
			int size=inputEvents[i].name.length();
			int j=0;
			for(;j<size;j++){
			allInputs[i].name[j]=inputEvents[i].name[j];	
			}
			allInputs[i].name[j]='\0';
			j=0;
			int size1=inputEvents[i].op.length();

			for(;j<size1;j++){
			allInputs[i].op[j]=inputEvents[i].op[j];	
			}
			allInputs[i].op[j]='\0';

			allInputs[i].value=inputEvents[i].value;
			
		}
		/*cout<<"\n\n"<<"InputEvents";
		for(int i=0;i<inputEvents.size();i++)
		{
			cout<<"\n"<<allInputs[i].name<<" "<<allInputs[i].op<<" "<<allInputs[i].value ;
		}*/

		newInput *d_allInputs;

		cudaMalloc((void **)&d_allInputs,inputEvents.size()*sizeof(struct newInput));		
		if(cudaSuccess!=cudaGetLastError())
		{
	        	printf("\nerror at cuda malloc sub");
		}
	
		cudaMemcpy(d_allInputs,allInputs,inputEvents.size()*sizeof(struct newInput),cudaMemcpyHostToDevice);
		if(cudaSuccess!=cudaGetLastError())
		{
	        	printf("\nerror at cuda mem cpy sub");
		}

		filter *d_allFilters;
		cudaMalloc((void **)&d_allFilters,(filters.size())*sizeof(struct filter));
		cudaMemcpy(d_allFilters,allFilters,filters.size()*sizeof(struct filter),cudaMemcpyHostToDevice);
		if(cudaSuccess!=cudaGetLastError())
		{
	        	printf("\nerror at cuda mem cpy sub");
		}
		
		//cudaEventCreate(&start_event) ;
		//cudaEventCreate(&stop_event) ;
		//cudaEventRecord(start_event, 0);
	
	        double iStart = seconds();
   		match<<<gridSize,dimBlock>>>(d_allConstraints,d_sizeC,d_allFilters,d_allInputs,maxConstraints);
	
		cudaThreadSynchronize();
	
    		if(cudaSuccess!=cudaGetLastError())
    		{
           	printf("\nerrorker:%s",cudaGetErrorString(cudaGetLastError()));
    		}   
		//printf("\npeek2:%s",cudaGetErrorString(cudaThreadSynchronize()));
		/*cudaEventRecord(stop_event, 0);
		cudaEventSynchronize( stop_event);
		cudaEventElapsedTime( &time, start_event, stop_event );
		cudaEventDestroy( start_event ); // cleanup
		cudaEventDestroy( stop_event );*/
		//printf("\ndone and it took: %f milliseconds\n", time);
                double iElaps = seconds() - iStart;
                printf("GPU timer elapsed: %8.2fms \n", iElaps * 1000.0);
		
		timefinal+=iElaps * 1000.0;
		//timefinal+=time;
	         //cout<<"time is : "<<time<<endl;
	        // exit(1);       
		//we want count of filter
    		cudaMemcpy(allFilters,d_allFilters,filters.size()*sizeof(struct filter),cudaMemcpyDeviceToHost);
		
		if(cudaSuccess!=cudaGetLastError())
    		{
         	  printf("\nerror3:%s",cudaGetErrorString(cudaGetLastError()));
    		}
                
                for(int x = 0 ; x < filters.size() ; x++ )
                 if(allFilters[x].count == allFilters[x].size)
		 cout<<e_list.at(i)<<" : "<<allFilters[x].count<<endl;  
		 //exit(1);
		

		for(int i=0;i<filters.size();i++)
		{
			//printf(" %d",filtersub[i].count);
			int res=allFilters[i].count/allFilters[i].size;
			finalres[i]+=res;
			results[index*MAX_LINES+i]+=res;
			allFilters[i].count=0;
			//printf("%d",finalres[i]);
		}
	/*	int count=0; 
		for(int i=0;i<cSize;i++)
	{
		if(results[i] >0)
		{
		        printf("\nResult-> %d %d ",i,results[i]);
	                count++;
	        }
	}*/
		
		//exit(1);
 		cudaFree(d_allInputs);
	cudaFree(d_allFilters);

	
	}//End of loop for input
		index++;
	//printf("\ndone and it took: %f milliseconds\n", timefinal);
	resultTime+=timefinal;
	cudaFree(d_allConstraints);
	cudaFree(d_sizeC);
	}
		
	//printf("\ndone and it took: %f milliseconds\n",resultTime);
        int count = 0;
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
	
	return 0;
}
