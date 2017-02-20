import panflute as pf


def getgitlink(elem_in, elem_out, linkstr=''):
    for i in elem_in.content:
        istr = pf.stringify(i)
        if linkstr or istr[:2] == '[[':
            linkstr += istr
            # is the link complete?
            if linkstr[-2:] == ']]':
                link = linkstr[2:-2].split('|')
                elem_out.content.append(pf.Link(pf.Str(link[0]), url=link[-1] + '.html'))
                linkstr = ''
        else:
            elem_out.content.append(i)

    return linkstr


def action(elem, doc):
    # Parse git links in a plain text line or a text paragraph
    if isinstance(elem, (pf.Para, pf.Plain)):
        newelem = elem.__class__()
        if getgitlink(elem, newelem) == '':
            return newelem
        else:
            return elem

    # Parse git links in a table row: fix problem where git links formatted
    # like [[text|link]] are spread over two table cells
    elif isinstance(elem, pf.TableRow):
        newelem = pf.TableRow()
        newcell = pf.TableCell()
        linkstr = ''
        for cell in elem.content:
            if len(newcell.content) == 0:
                newcell.content.append(cell.content[0].__class__())
            if len(linkstr) > 0:
                linkstr += '|'
            linkstr = getgitlink(cell.content[0], newcell.content[0], linkstr)
            if not linkstr:
                newelem.content.append(newcell)
                newcell = pf.TableCell()

        return newelem


def prepare(doc):
    pass


def finalize(doc):
    pass


def main(doc=None):
    return pf.run_filter(action,
                         prepare=prepare,
                         finalize=finalize,
                         doc=doc)


if __name__ == '__main__':
    main()
