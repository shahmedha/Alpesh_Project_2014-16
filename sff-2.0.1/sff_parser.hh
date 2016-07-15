/* A Bison parser, made by GNU Bison 3.0.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2013 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

#ifndef YY_YY_SFF_PARSER_HH_INCLUDED
# define YY_YY_SFF_PARSER_HH_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token type.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    ID_v = 258,
    STR_v = 259,
    REGEX_V = 260,
    INT_v = 261,
    BOOL_v = 262,
    DOUBLE_v = 263,
    AND_op = 264,
    OR_op = 265,
    LT_op = 266,
    GT_op = 267,
    EQ_op = 268,
    NE_op = 269,
    PF_op = 270,
    SF_op = 271,
    SS_op = 272,
    RE_op = 273,
    INTEGER_kw = 274,
    STRING_kw = 275,
    BOOLEAN_kw = 276,
    DOUBLE_kw = 277,
    ANY_kw = 278,
    IFCONFIG_kw = 279,
    SELECT_kw = 280,
    SET_kw = 281,
    CONSOLIDATE_kw = 282,
    OUTPUT_kw = 283,
    STATISTICS_kw = 284,
    CLEAR_kw = 285,
    TIMER_kw = 286,
    HELP_kw = 287
  };
#endif
/* Tokens.  */
#define ID_v 258
#define STR_v 259
#define REGEX_V 260
#define INT_v 261
#define BOOL_v 262
#define DOUBLE_v 263
#define AND_op 264
#define OR_op 265
#define LT_op 266
#define GT_op 267
#define EQ_op 268
#define NE_op 269
#define PF_op 270
#define SF_op 271
#define SS_op 272
#define RE_op 273
#define INTEGER_kw 274
#define STRING_kw 275
#define BOOLEAN_kw 276
#define DOUBLE_kw 277
#define ANY_kw 278
#define IFCONFIG_kw 279
#define SELECT_kw 280
#define SET_kw 281
#define CONSOLIDATE_kw 282
#define OUTPUT_kw 283
#define STATISTICS_kw 284
#define CLEAR_kw 285
#define TIMER_kw 286
#define HELP_kw 287

/* Value type.  */



int yyparse (void);

#endif /* !YY_YY_SFF_PARSER_HH_INCLUDED  */
