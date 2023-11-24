import torch
from transformers import BartForConditionalGeneration, BartTokenizer

def summarize_text(text, max_length=1000, model_name="facebook/bart-large-cnn"):
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs, max_length=max_length, min_length=500, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
text = input("Enter The paragraph for summarization : ");


text="""
Functions
Functions are one of the fundamental building blocks of C++. The FIRST program consists
almost entirely of a single function called main(). The only parts of this program that are not
part of the function are the first two lines—the ones that start with #include and using. (We’ll
see what these lines do in a moment.)
We noted in Chapter 1, “The Big Picture,” that a function can be part of a class, in which case
it is called a member function. However, functions can also exist independently of classes. We
are not yet ready to talk about classes, so we will show functions that are separate standalone
entities, as main() is here.
Function Name
The parentheses following the word main are the distinguishing feature of a function. Without
the parentheses the compiler would think that main refers to a variable or to some other program
element. When we discuss functions in the text, we’ll follow the same convention that
C++ uses: We’ll put parentheses following the function name. Later on we’ll see that the
parentheses aren’t always empty. They’re used to hold function arguments: values passed from
the calling program to the function.
The word int preceding the function name indicates that this particular function has a return
value of type int. Don’t worry about this now; we’ll learn about data types later in this chapter
and return values in Chapter 5, “Functions.”
Braces and the Function Body
The body of a function is surrounded by braces (sometimes called curly brackets). These
braces play the same role as the BEGIN and END keywords in some other languages: They surround
or delimit a block of program statements. Every function must use this pair of braces
around the function body. In this example there are only two statements in the function body:
the line starting with cout, and the line starting with return. However, a function body can
consist of many statements.
Always Start with main()
When you run a C++ program, the first statement executed will be at the beginning of a function
called main(). (At least that’s true of the console mode programs in this book.) The program
may consist of many functions, classes, and other program elements, but on startup,
control always goes to main(). If there is no function called main() in your program, an error
will be reported when you run the program.
In most C++ programs, as we’ll see later, main() calls member functions in various objects to
carry out the program’s real work. The main() function may also contain calls to other standalone
functions. This is shown in Figure 2.1.
C++ Programming Basics
2
C++
PROGRAMMING
BASICS
"""

summarized_text = summarize_text (text)
print("Summarized  Version is :")
print(summarized_text)
