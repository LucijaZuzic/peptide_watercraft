from datetime import date

today = date.today()

file_bibliography = open("sn-bibliography.bib", "r")
all_rows_bibliography = file_bibliography.readlines()
file_bibliography.close()

file_tex = open("sn-article.tex", "r", encoding = "UTF8")
all_rows_tex = file_tex.readlines()
all_tex = ""
for line in all_rows_tex:
    all_tex += line
    if "\csname PreBibitemsHook\endcsname" in line:
        break
file_tex.close()

unused_fields = set()
used_fields = {"pages", "language", "month", "year", "author", 
               "address", "volume", "issue", "doi", "url", 
               "title", "article", "journal", "editor", "publisher",
               "issn", "isbn", "eprint", "archiveprefix"
               }
entries = dict()
for line in all_rows_bibliography:
    if "@" in line:
        ix_key = line.find("{")
        key = line[ix_key+1:line.find(",")]
        type_key = line[1:ix_key]
    else:
        if "=" not in line:
            ln = line.replace("}", "").replace("{", "").strip()
            if len(ln):
                print("Empty lin", ln)
            continue
        ix_equal = line.find("=")
        category = line[:ix_equal].strip()
        value = line[ix_equal+1:].strip()
        if "{" in value:
            if value[-1] == ",":
                value = value[1:-2]
            else:
                value = value[1:-1]
        else:
            if '"' == value[0]:
                #print("Apostrophe lin", key, value, category)
                if value[-1] == ",":
                    value = value[1:-2]
                else:
                    value = value[1:-1]
                #print("Apostrophe lin", key, value, category)
            else:
                #print("No delimiter lin", key, value, category)
                if value[-1] == ",":
                    value = value[:-1]
                #print("delimiter lin", key, value, category)
    if key not in entries:
        entries[key] = {"type_of_entry": type_key.lower(), "position_tex": all_tex.find(key)}
    else:
        entries[key][category.lower()] = value
        if category.lower() not in used_fields:
            unused_fields.add(category.lower())

list_key = [entries[key]["type_of_entry"] for key in entries]
set_key = set([entries[key]["type_of_entry"] for key in entries])
count_key = {k: list_key.count(k) for k in set_key}

locations_keys = {entries[key]["position_tex"]: key for key in entries}
sorted_loc = sorted([entries[key]["position_tex"] for key in entries])

for key in entries:
    if entries[key]["position_tex"] == -1:
        print("NO POS", key)
        continue
    if entries[key]["type_of_entry"] in ["article"]:#in ["article", "inproceedings"]:
        if "doi" not in entries[key]:
            print(key, "no DOI")
    if entries[key]["type_of_entry"] in ["inbook", "book", "incollection", "inproceedings"]:
        if "address" not in entries[key]:
            print(key, "no address")
        if "publisher" not in entries[key]:
            print(key, "no publisher")
        if "publisher" in entries[key] and "address" not in entries[key]:
            print(key, "no address for publisher")
    if "doi" in entries[key] and "url" not in entries[key]:
        print(key, "no DOI url")
        entries[key]["url"] = "http://dx.doi.org/" + entries[key]["doi"]
    if "doi" in entries[key] and "url" in entries[key]:
        if entries[key]["url"] != "http://dx.doi.org/" + entries[key]["doi"]:                    
            print(key, "wrong url", entries[key]["url"], entries[key]["doi"])

no_volume = set()
no_issue = set()
no_pages = set()
no_pages_book = set()
no_editor_book = set()
editor_book = set()

ix = 0
strtotal = ""
for position_key in sorted_loc:
    if position_key == -1:
        print("NO POS", position_key, locations_keys[position_key])
        continue
    ix += 1
    strpr = "%%% " + str(ix) + "\n"
    strpr += "\\bibitem{" + locations_keys[position_key] + "}\n"
    type_entry = entries[locations_keys[position_key]]["type_of_entry"]
    if type_entry == "incollection" or type_entry == "inbook" or type_entry == "inproceedings":
        type_entry = "chapter"
    if type_entry != "article" and type_entry != "book" and type_entry != "chapter":
        type_entry = "otherref"
        print(locations_keys[position_key], "otherref")
    strpr += "\\begin{b" + type_entry + "}\n"
    if "author" not in entries[locations_keys[position_key]]:
        entries[locations_keys[position_key]]["author"] = "RDevelopers, R."
        print("RDevelopers", locations_keys[position_key])
    if "year" not in entries[locations_keys[position_key]]:
        entries[locations_keys[position_key]]["year"] = "1901"
        print("1901", locations_keys[position_key])
    authors_list = entries[locations_keys[position_key]]["author"].split(" and ")
    authors_list = [author.split(",") for author in authors_list]
    authors_list = [[author_part.strip().replace(". ", ".") for author_part in author_surname_firstname] for author_surname_firstname in authors_list]
    for author_surname_firstname in authors_list:
        if len(author_surname_firstname) > 1:
            if not len(author_surname_firstname[1]) != author_surname_firstname[1].count(".") * 2 - 1 or not author_surname_firstname[1][-1] == ".":
                print("Author initial err", locations_keys[position_key],entries[locations_keys[position_key]]["author"] )
            strpr += "\\bauthor{\\bsnm{" + author_surname_firstname[0] + "}, \\binits{" + author_surname_firstname[1] + "}}"
        else:
            strpr += "\\bauthor{\\bsnm{" + author_surname_firstname[0] + "}}"
        if author_surname_firstname == authors_list[-1]:
            strpr += "\n"
        else:
            if author_surname_firstname == authors_list[-2]:
                strpr += ", \\&\n"
            else:
                strpr += ",\n"
    strpr += "(\\byear{" + entries[locations_keys[position_key]]["year"] + "}).\n"
    title_new = entries[locations_keys[position_key]]["title"].replace("{", "").replace("}", "")
    replaced_letters = set()
    for letter in entries[locations_keys[position_key]]["title"]:
        if letter.isupper() and letter not in replaced_letters:
            title_new = title_new.replace(letter, "{" + letter + "}")
            replaced_letters.add(letter)
    replaced_letters = set()
    for letter_ix in range(len(title_new) - 2):
        letter = title_new[letter_ix]
        letter_next = title_new[letter_ix + 1]
        letter_next_next = title_new[letter_ix + 2]
        if letter == "\\" and letter_next.isalpha() and letter_next_next != "{" and letter + letter_next + letter_next_next not in replaced_letters:
            print(title_new, letter + letter_next + letter_next_next, "{" + letter + letter_next + "{" + letter_next_next + "}}")
            title_new = title_new.replace(letter + letter_next + letter_next_next, "{" + letter + letter_next + "{" + letter_next_next + "}}")
            print(title_new)
            replaced_letters.add(letter + letter_next + letter_next_next)
    if type_entry == "otherref":
        strpr = strpr.replace("bauthor", "oauthor")
        strpr += title_new + ".\n" 
    if type_entry == "article":
        strpr += "\\batitle{" + title_new + "}.\n"               
        strpr += "\\bjtitle{" + entries[locations_keys[position_key]]["journal"] + "}"   
        if "volume" not in entries[locations_keys[position_key]] and "pages" not in entries[locations_keys[position_key]]:     
            strpr += ".\n"
        else:
            strpr += ",\n"
        if "volume" in entries[locations_keys[position_key]]:          
            strpr += "\\bvolume{" + entries[locations_keys[position_key]]["volume"] + "}" 
            if "issue" in entries[locations_keys[position_key]]:          
                strpr += "(\\bissue{" + entries[locations_keys[position_key]]["issue"] + "})" 
            else:
                no_issue.add(locations_keys[position_key])
            if "pages" not in entries[locations_keys[position_key]]:     
                strpr += ".\n"
            else:
                strpr += ",\n"
        else:
            no_volume.add(locations_keys[position_key])
        if "pages" in entries[locations_keys[position_key]]:
            pages_new = entries[locations_keys[position_key]]["pages"].replace("--", "-").replace(" ", "").split("-")
            if len(pages_new) > 1:           
                strpr += "\\bfpage{" + pages_new[0] + "}--\\bfpage{" + pages_new[1] + "}.\n"   
            else:   
                strpr += "\\bfpage{" + pages_new[0] + "}.\n" 
        else:
            no_pages.add(locations_keys[position_key])
    if type_entry == "book" or type_entry == "chapter":
        if type_entry == "chapter":
            if "booktitle" not in entries[locations_keys[position_key]]:
                entries[locations_keys[position_key]]["booktitle"] = "nobooktitle"
                print("nobooktitle", locations_keys[position_key])
            book_title_new = entries[locations_keys[position_key]]["booktitle"].replace("{", "").replace("}", "")
            replaced_letters = set()
            for letter in entries[locations_keys[position_key]]["booktitle"]:
                if letter.isupper() and letter not in replaced_letters:
                    book_title_new = book_title_new.replace(letter, "{" + letter + "}")
                    replaced_letters.add(letter)      
            strpr += "\\bctitle{" + title_new + "}.\n"   
        if "pages" in entries[locations_keys[position_key]] or type_entry == "chapter":
            strpr += "In "
        if "editor" in entries[locations_keys[position_key]]:
            editors_list = entries[locations_keys[position_key]]["editor"].split(" and ")
            editors_list = [editor.split(",") for editor in editors_list]
            editors_list = [[editor_part.strip().replace(". ", ".") for editor_part in editor_surname_firstname] for editor_surname_firstname in editors_list]
            for editor_surname_firstname in editors_list:
                if len(editor_surname_firstname) > 1:
                    strpr += "\\beditor{\\binits{" + editor_surname_firstname[1] + "} \\bsnm{" + editor_surname_firstname[0] + "}}"
                else:
                    strpr += "\\beditor{\\bsnm{" + editor_surname_firstname[0] + "}}"
                if editor_surname_firstname == editors_list[-1]:
                    strpr += " (Ed.),\n"
                else:
                    if editor_surname_firstname == editors_list[-2]:
                        strpr += ", \\&\n"
                    else:
                        strpr += ",\n"
            editor_book.add(locations_keys[position_key])
        else:
            no_editor_book.add(locations_keys[position_key])   
        if type_entry == "chapter":    
            strpr += "\\bbtitle{" + book_title_new + "}"  
        else:
            strpr += "\\bbtitle{" + title_new + "}"  
        if "pages" in entries[locations_keys[position_key]]:
            strpr += "\n" 
            pages_new = entries[locations_keys[position_key]]["pages"].replace("--", "-").replace(" ", "").split("-")
            if len(pages_new) > 1:           
                strpr += "(pp. \\bfpage{" + pages_new[0] + "}--\\bfpage{" + pages_new[1] + "}).\n"   
            else:   
                strpr += "(pp. \\bfpage{" + pages_new[0] + "}).\n" 
        else:
            strpr += ".\n" 
            no_pages_book.add(locations_keys[position_key])
        if "publisher" in entries[locations_keys[position_key]]:
            if "address" in entries[locations_keys[position_key]]:
                strpr += "\\blocation{" + entries[locations_keys[position_key]]["address"] + "}: \n"  
            strpr += "\\bpublisher{" + entries[locations_keys[position_key]]["publisher"] + "}.\n"
    if "doi" in entries[locations_keys[position_key]]:
        strpr += "doi:" + entries[locations_keys[position_key]]["doi"] + ".\n"
        #strpr += "\\doiurl{" + entries[locations_keys[position_key]]["doi"] + "}.\n"
    if "note" in entries[locations_keys[position_key]] and "url" in entries[locations_keys[position_key]]:
        strpr += "\\url{" + entries[locations_keys[position_key]]["url"] + "}.\n"
        strpr += "Accessed " + today.strftime("%d %B %Y") + ".\n"
    strpr += "\\end{b" + type_entry + "}\n"
    strpr += "\\endbibitem\n"
    strtotal += strpr + "\n"
        
file_bibliography = open("sn-bibliography-new.txt", "w")
file_bibliography.write(strtotal[:-2])
file_bibliography.close()

print(count_key, len(list_key))

reordered = ""

for position_key in sorted_loc:
    if position_key == -1:
        print("NO POS", position_key, locations_keys[position_key])
        continue
    reorderedone = "@" + entries[locations_keys[position_key]]["type_of_entry"] + "{" + locations_keys[position_key] + ",\n"
    for k in entries[locations_keys[position_key]]:
        if k == "type_of_entry" or k == "position_tex":
            continue
        value = entries[locations_keys[position_key]][k]
        if "title" == k:
            title_new = value.replace("{", "").replace("}", "")
            replaced_letters = set()
            for letter in entries[locations_keys[position_key]][k]:
                if letter.isupper() and letter not in replaced_letters:
                    title_new = title_new.replace(letter, "{" + letter + "}")
                    replaced_letters.add(letter)
            replaced_letters = set()
            for letter_ix in range(len(title_new) - 2):
                letter = title_new[letter_ix]
                letter_next = title_new[letter_ix + 1]
                letter_next_next = title_new[letter_ix + 2]
                if letter == "\\" and letter_next.isalpha() and letter_next_next != "{" and letter + letter_next + letter_next_next not in replaced_letters:
                    print(title_new, letter + letter_next + letter_next_next, "{" + letter + letter_next + "{" + letter_next_next + "}}")
                    title_new = title_new.replace(letter + letter_next + letter_next_next, "{" + letter + letter_next + "{" + letter_next_next + "}}")
                    print(title_new)
                    replaced_letters.add(letter + letter_next + letter_next_next)
            value = title_new
        if "pages" == k:
            pages_new = entries[locations_keys[position_key]][k].replace("--", "-").replace(" ", "").split("-")
            if len(pages_new) > 1:           
                value = pages_new[0] + "--" + pages_new[1]
            else:   
                value = pages_new[0]
        reorderedone += "\t" + k + " = {" + value + "},\n"
    reorderedone = reorderedone[:-2] + "\n}\n\n"
    reordered += reorderedone
#print(reordered[:-2])

file_bibliographyn = open("sn-bibliography-new-new.txt", "w")
file_bibliographyn.write(reordered[:-2])
file_bibliographyn.close()

file_bibliographyart = open("sn-article-new.txt", "w", encoding = "UTF8")
file_bibliographyart.write(all_tex.replace("\csname PreBibitemsHook\endcsname\n", "\csname PreBibitemsHook\endcsname\n\n" + strtotal[:-2] + "\n\n\n\\end{thebibliography}\n\n\n\\end{document}"))
file_bibliographyart.close()