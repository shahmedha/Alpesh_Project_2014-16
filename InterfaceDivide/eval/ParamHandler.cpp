#include "ParamHandler.h"

ParamHandler::ParamHandler() {
	setDefaultValues();
}

ParamHandler::~ParamHandler() {
	// Nothing to do
}

void ParamHandler::resetToDefault() {
	setDefaultValues();
}

int ParamHandler::getNumMessages() const {
	return numMessages;
}

int ParamHandler::getMaxAttr() const {
	return maxAttr;
}

int ParamHandler::getMaxConstr() const {
	return maxConstr;
}

int ParamHandler::getMaxFilters() const {
	return maxFilters;
}

int ParamHandler::getMinAttr() const {
	return minAttr;
}

int ParamHandler::getMinConstr() const {
	return minConstr;
}

int ParamHandler::getMinFilters() const {
	return minFilters;
}

int ParamHandler::getNumAttVal() const {
	return numAttVal;
}

int ParamHandler::getNumIf() const {
	return numIf;
}

int ParamHandler::getNumIntNames() const {
	return numIntNames;
}

int ParamHandler::getNumNames() const {
	return numNames;
}

int ParamHandler::getConstraintsPerThread() const {
	return contraintsPerThread;
}

void ParamHandler::setNumMessages(int numMessages) {
	this->numMessages = numMessages;
}

void ParamHandler::setMaxAttr(int maxAttr) {
	this->maxAttr = maxAttr;
}

void ParamHandler::setMaxConstr(int maxConstr) {
	this->maxConstr = maxConstr;
}

void ParamHandler::setMaxFilters(int maxFilters) {
	this->maxFilters = maxFilters;
}

void ParamHandler::setMinAttr(int minAttr) {
	this->minAttr = minAttr;
}

void ParamHandler::setMinConstr(int minConstr) {
	this->minConstr = minConstr;
}

void ParamHandler::setMinFilters(int minFilters) {
	this->minFilters = minFilters;
}

void ParamHandler::setNumAttVal(int numAttVal) {
	this->numAttVal = numAttVal;
}

void ParamHandler::setNumIf(int numIf) {
	this->numIf = numIf;
}

void ParamHandler::setNumIntNames(int numIntNames) {
	this->numIntNames = numIntNames;
}

void ParamHandler::setNumNames(int numNames) {
	this->numNames = numNames;
}

void ParamHandler::setConstraintsPerThread(int constraintsPerThread) {
	this->contraintsPerThread = constraintsPerThread;
}

int ParamHandler::getPercIntDf() {
	return percIntDf;
}

int ParamHandler::getPercIntEq() {
	return percIntEq;
}

int ParamHandler::getPercIntGt() {
	return percIntGt;
}

int ParamHandler::getPercIntLt() {
	return percIntLt;
}

int ParamHandler::getPercStrDf() {
	return percStrDf;
}

int ParamHandler::getPercStrEq() {
	return percStrEq;
}

int ParamHandler::getPercStrIn() {
	return percStrIn;
}

int ParamHandler::getPercStrPf() {
	return percStrPf;
}

bool ParamHandler::getZipfNames() {
	return zipfNames;
}

void ParamHandler::setPercIntDf(int percIntDf) {
	this->percIntDf = percIntDf;
}

void ParamHandler::setPercIntEq(int percIntEq) {
	this->percIntEq = percIntEq;
}

void ParamHandler::setPercIntGt(int percIntGt) {
	this->percIntGt = percIntGt;
}

void ParamHandler::setPercIntLt(int percIntLt) {
	this->percIntLt = percIntLt;
}

void ParamHandler::setPercStrDf(int percStrDf) {
	this->percStrDf = percStrDf;
}

void ParamHandler::setPercStrEq(int percStrEq) {
	this->percStrEq = percStrEq;
}

void ParamHandler::setPercStrIn(int percStrIn) {
	this->percStrIn = percStrIn;
}

void ParamHandler::setPercStrPf(int percStrPf) {
	this->percStrPf = percStrPf;
}

void ParamHandler::setZipfNames(bool zipfNames) {
	this->zipfNames = zipfNames;
}

void ParamHandler::setDefaultValues() {
	numMessages = 1000;
	numIf = 10;
	minFilters = 22500;
	maxFilters = 27500;
	minConstr = 3;
	maxConstr = 5;
	minAttr = 3;
	maxAttr = 5;
	numAttVal = 100;
	numNames = 100;
	numIntNames = 100;
	percIntEq = 25;
	percIntDf = 25;
	percIntGt = 25;
	percIntLt = 25;
	percStrEq = 25;
	percStrDf = 25;
	percStrIn = 25;
	percStrPf = 25;
	zipfNames = false;
}
