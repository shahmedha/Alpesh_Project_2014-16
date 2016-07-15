// -*-C++-*-
//
//  This file is part of Siena, a wide-area event notification system.
//  See http://www.inf.usi.ch/carzaniga/siena/
//
//  Author: Antonio Carzaniga
//  See the file AUTHORS for full details. 
//
//  Copyright (C) 1998-2003 University of Colorado
//
//  Siena is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Siena is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Siena.  If not, see <http://www.gnu.org/licenses/>.
//
#ifndef _YYSFFTYPES_H
#define _YYSFFTYPES_H

#include <string>

#include <siena/attributes.h>

#include "simple_attributes_types.h"
#include "simple_tags_types.h"

struct sffstype {
    const std::string *		fname;
    unsigned			nlin;
    union {
	siena::Int		int_v;
	siena::Double		double_v;
	siena::Bool		bool_v;
	std::string *		str_v;
	siena::OperatorId	op;
	simple_message *	message;
	simple_value *		value;
	simple_op_value *	constraint;
	simple_predicate *	predicate;
	simple_filter *		filter;
	simple_tagset *		tagset;
	simple_tagset_list *	tagset_list;
    };
};

typedef struct sffstype YYSTYPE;

#define YY_DECL int yylex(YYSTYPE * yylval)
extern YY_DECL;

extern int yylineno;
extern std::string yysfffname;

#endif
