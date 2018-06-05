#!/bin/bash

# get all files
rm -f imagegenerator.*
shopt -s nullglob
files=(*.tex)

#prepare output
mkdir -p PNG

ctr=1;
for i in "${files[@]}"; do 
  rm imagegenerator.*

  echo "
\documentclass[preview,border=4mm]{standalone}
\usepackage{url}
\usepackage{tikz}
\usepackage{color}
\usepackage{float}
\usepackage{multirow}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{amsfonts}
\usepackage{amsthm}
\usepackage{IEEEtrantools}
\usepackage[shortlabels]{enumitem}
\usepackage{arydshln}

\usetikzlibrary{fit,
            backgrounds,
            arrows,
            decorations.markings,
            decorations.pathmorphing,
            snakes,
            shapes.misc, 
            positioning,
            scopes}

%quantum commands
\renewcommand{\H}{\mathcal{H}}
\newcommand{\ket}[1]{|#1 \rangle}
\newcommand{\bra}[1]{\langle #1|}
\newcommand{\braket}[2]{\langle#1|#2\rangle}
\newcommand{\bk}[2]{\langle#1|#2\rangle}
\newcommand{\ketbra}[2]{|#1 \rangle\langle #2|}
\newcommand{\kb}[2]{|#1 \rangle\langle #2|}
\newcommand{\proj}[1]{|#1\rangle\langle #1|}

% Math environments
\theoremstyle{remark}
\newtheorem{remark}{Remark}
\newtheorem{xmpl}{Example}

%Math commands
\DeclareMathOperator{\Tr}{Tr} % Trace
\DeclareMathOperator{\Ent}{H} % Entropy
\DeclareMathOperator{\I}{I} % Mutual Information
\newcommand{\keyrate}[3]{S(#1;#2||#3)} % secret key rate
\newcommand{\intrinfo}[3]{\I(#1;#2\downarrow#3)} % intrinsic information
\newcommand{\redintrinfo}[3]{\I(#1;#2\downdownarrows#3)} %reduced intrinsic information

  " | tee -a imagegenerator.tex

  echo "
\tikzset{colorbox/.style={thick, rounded corners=2pt, text height=1.7ex,text depth=.25ex, draw=#1!70!black, fill=#1!30}}
\tikzset{colorelement/.style={thick, rounded corners=2pt, draw=#1!70!black, fill=#1!30}}
\tikzset{conn/.style={thick, shorten <=#1, shorten >=#1}}
\tikzset{arr_node/.style={pos=0.5,above,font=\scriptsize, sloped}}
  " | tee -a imagegenerator.tex

  echo "
 \begin{document}
  " | tee -a imagegenerator.tex


  echo "\\begin{figure}" | tee -a imagegenerator.tex
  echo "\\centering" | tee -a imagegenerator.tex
  echo "\\input{${i/\.tex/}}" | tee -a imagegenerator.tex
  # echo "\\caption{${i//_/\\_}}" | tee -a imagegenerator.tex
  echo "\\end{figure}" | tee -a imagegenerator.tex
  echo "

  " | tee -a imagegenerator.tex

  echo "\\end{document}" | tee -a imagegenerator.tex

  pdflatex imagegenerator.tex
  pdflatex imagegenerator.tex

  mv "imagegenerator.pdf" "PNG/${i/\.tex/}.pdf"

  sips -s format png "PNG/${i/\.tex/}.pdf" --out "PNG/${i/\.tex/}.png"

  # rm imagegenerator.*

  ((ctr++))
done
