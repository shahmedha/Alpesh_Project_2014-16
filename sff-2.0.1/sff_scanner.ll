/*
 *  This file is part of Siena, a wide-area event notification system.
 *  See http://www.inf.usi.ch/carzaniga/siena/
 *  
 *  Author: Antonio Carzaniga
 *  See the file AUTHORS for full details. 
 *  
 *  Copyright (C) 1998-2003 University of Colorado
 *  
 *  Siena is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *  
 *  Siena is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *  
 */
%option yylineno
%option nounput
%option interactive
%option always-interactive
%{
#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#if HAVE_UNISTD_H
#include <unistd.h>
#else
int isatty(int i) {
    return 0
}
#endif

#include <string>
#include <cstdlib>

#include "yysfftypes.h"
/* "yysfftypes.h" must be #included before "sff_parser.h" */
#include "sff_parser.hh"

#define YY_NO_UNPUT

std::string yysfffname = ""; /* Current source file name */
extern int yyerror (const char *);

extern void sff_parser_prompt();
extern void sff_parser_incomplete_command();

static int yyin_is_interactive = -1;

int sff_scanner_is_interactive() {
    if (yyin_is_interactive == -1) {
	if (!yyin) 
	    yyin = stdin;
        int fd = fileno(yyin);
	if (fd < 0) {
	    yyin_is_interactive = 0;
	} else {
	    yyin_is_interactive = isatty(fd);
	}
    }
    return yyin_is_interactive;
}

int sff_scanner_open(const char * fname) {
    if (fname != 0 && strcmp (fname, "-")) {
        yysfffname = fname;
        yyin = fopen(fname, "r");
	yyin_is_interactive = 0;
    } else {
	yysfffname = "{standard input}";
        yyin = stdin;
	yyin_is_interactive = -1;
    }
    yylineno = 1;
    return yyin == NULL;
}

int sff_scanner_close() {
    if (yyin != NULL && yyin != stdin) {
	int res = fclose(yyin);
	yyin = NULL;
	return res;
    } else 
	return 0;
}

#define RET(x) {yylval->nlin = yylineno; yylval->fname = & yysfffname;sff_parser_incomplete_command();};return(x)

%}

space			[ \b\t\v\f\r]+
alpha			[a-zA-Z_]
nonzero			[1-9]
digit			[0-9]
oct_digit		[0-7]
hex_digit		[0-9a-fA-F]
ident_extra		[/.\-]
ident			{alpha}({digit}|{alpha}|{ident_extra})*
sign_opt		[\-+]?
decimal			({sign_opt}{nonzero}{digit}*)|("0")
octal			{oct_digit}{1,3}
strhex			"0"?[Xx]{hex_digit}+
hex			"0"[Xx]{hex_digit}+
exponent		[eE]{sign_opt}{digit}+
fractional		({digit}+".")|({digit}*"."{digit}+)
fpoint			{sign_opt}{fractional}{exponent}?
u			[Uu]
l			[Ll]
f			[Ff]

/* Parts of a regular expressions. not_bracket is simply "everything but
   a bracket", but flex versions > 2.5.4 do not like two consecutive
   brackets in the input file!

   When parsing a regex, we don't want the / to be a terminator if
   it is within a bracket, i.e. /[abc/def]/ must be a valid regex.
   This is like sed, but unlike perl.  For this reason, we define one
   start conditions for brackets, that eats a 'bracket_elem' at a time.
   Note that a bracket is started with "["\^?\]? in order to eat the
   initial caret and a literal closing bracket.

   Also note that we want to parse escapes differently in strings and
   regexes, which is why we need a different token:

   0) Of course the above handling of brackets does not belong in strings.
   1) \b (bell, ASCII 7) is not supported in regexes.
   2) Unrecognized escapes (e.g. \< or \b itself) must be passed down
      to the regex matcher, rather than stripped. 
   3) We give an error if a regex is multiline unless each line is
      \-terminated.  In the future this may be extended to strings.

   coll_class is something like Perl's \[: .*? :\] and it can be
   "dissected" this way:

	"[:"			begin the sequence
	(\]|			closing bracket not preceded by :
	\:{not_bracket}|	or : not followed by closing bracket
	[^\]:])*		or any other char -- ad libitum
	":"+\]			and the end of the sequence		*/

not_bracket		(a|[^\]a])
coll_class		"[:"(\]|\:{not_bracket}|[^\]:])*":"+\]
coll_elem		"[."(\]|\.{not_bracket}|[^\].])*"."+\]
coll_equiv		"[="(\]|\={not_bracket}|[^\]=])*"="+\]
bracket_elem		({not_bracket}|{coll_elem}|{coll_class}|{coll_equiv})

/* 
 *  mutually exclusive states 
 */
%x comment strlit regex bracket

%% 

^#.*\n			{ const char * nlin; const char * fname;
                          if ((nlin = strtok(yytext, "\n#\t")) != NULL &&
                              (fname = strtok(NULL, " \t\"")) != NULL) {
			      yylineno = strtol(nlin, NULL, 10);
			      yysfffname = fname;
                          }
                        }

"/*"			BEGIN(comment);
<comment>[^\*\n]*	;
<comment>\*[\*]*[^/\n]	;
<comment>\n		;
<comment>[\*]*"*/"	BEGIN(INITIAL);

"//"[^\n]*\n		;		

{space}			; 
\n			{ if (sff_scanner_is_interactive()) 
			      sff_parser_prompt(); };

\"			{ yylval->str_v = new std::string(); BEGIN(strlit); }
"/"			{ yylval->str_v = new std::string(); BEGIN(regex); }

<strlit,regex,bracket>{
  /* More specific rules come first, so that they win over the generic "\\."
     rule below.  */
  \\n		*yylval->str_v += '\n';
  \\t		*yylval->str_v += '\t';
  \\r		*yylval->str_v += '\r';
  \\e		*yylval->str_v += '\e';
  \\f		*yylval->str_v += '\f';
  \\a		*yylval->str_v += '\a';
  \\v		*yylval->str_v += '\v';
  \\{octal}	{ char *c = yytext + 1; /* skip '\\' */
                  *yylval->str_v += (char)(strtoul(c, NULL, 8) % 256); }
  \\{strhex}	{ char *c = yytext + (yytext[1] == '0' ? 3 : 2); /* skip "\\0x" */
                  *yylval->str_v += (char)(strtoul(c, NULL, 16) % 256); }
}

<strlit>{
  \\b		*yylval->str_v += '\b';
  \n		{ *yylval->str_v += '\n'; }
  \"		{ BEGIN(INITIAL); RET(STR_v); }
  [^\"\\\n]+	{ yylval->str_v->append(yytext); }
  \\\?		*yylval->str_v += '\?';
  \\.		*yylval->str_v += yytext[1];
}

<regex,bracket>{
  /* A regex is very similar to a string literal, but unrecognized escapes
     are passed to the regex parser for further interpretation.  In addition,
     \cX sequences are recognized to mean Ctrl-X.  */
  \\c[a-z]		    *yylval->str_v += (char) (yytext[2] ^ 96);
  \\c[?-_]		    *yylval->str_v += (char) (yytext[2] ^ 64);
  \\.			    { yylval->str_v->append(yytext); }
  \n			    { yyerror ("new-line in regex"); BEGIN(INITIAL);
			      RET (REGEX_V); }
}

<regex>{
  "/"			    { BEGIN(INITIAL); RET(REGEX_V); }
  [^[/\\\n]+		    { yylval->str_v->append(yytext); }
  \[\^?\]?		    { yylval->str_v->append(yytext); BEGIN (bracket); }
}

<bracket>{
  \]			    { yylval->str_v->append(yytext); BEGIN(regex); }
  {bracket_elem}	    { yylval->str_v->append(yytext); }
}

","			RET(AND_op);
"|"			RET(OR_op);
"&&"			RET(AND_op);
"||"			RET(OR_op);
"="			RET(EQ_op);
">"			RET(GT_op);
"<"			RET(LT_op);
"!="			RET(NE_op);
"=*"			RET(PF_op);
"*="			RET(SF_op);
"**"			RET(SS_op);
"~="			RET(RE_op);

"ifconfig"		RET(IFCONFIG_kw);
"select"		RET(SELECT_kw);
"set"			RET(SET_kw);
"consolidate"		RET(CONSOLIDATE_kw);
"output"		RET(OUTPUT_kw);
"statistics"		RET(STATISTICS_kw);
"clear"			RET(CLEAR_kw);
"timer"			RET(TIMER_kw);
"help"			RET(HELP_kw);

"true"			{ yylval->bool_v = true; RET(BOOL_v); }
"false"			{ yylval->bool_v = false; RET(BOOL_v); }

"any"			RET(ANY_kw);
"integer"		RET(INTEGER_kw);
"string"		RET(STRING_kw);
"boolean"		RET(BOOLEAN_kw);
"double"		RET(DOUBLE_kw);

{ident}			{ yylval->str_v = new std::string(yytext); RET(ID_v); }

{decimal}		{ yylval->int_v = strtoll(yytext, NULL, 10); RET(INT_v); }

{hex}			{ yylval->int_v = strtoul(yytext, NULL, 16); RET(INT_v); }

{fpoint}		{ yylval->double_v = strtod(yytext, NULL); RET(DOUBLE_v); }

.			RET(*yytext);

%%

int yywrap() { return 1; }
