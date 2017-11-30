def splitarg(line):
    """
    Returns: a list with arguments and a list their starting locations
             in orginal line
    
    Break up a line in separate arguments, where:
    - comma is one separator
    - multiple commas are more separators
    - several spaces is one separator
    - one comma with leading/trailing space(s) is also one separator
    """

    # Create result lists
    args     = []
    icolarg  = []   # column wehere arg started

    # Process line
    templine = line.rstrip() # local copy to reduce
    idx = len(templine)
    while idx>0:
        idx = max(templine.rfind(","),templine.rfind(" "))
        args.append(templine[idx+1:].strip())
        icolarg.append(idx+1)
        templine = templine[:max(0,idx+1)].rstrip()
        if idx>0 and templine[-1]==",":
            templine = templine[:-1].rstrip()
    return args[::-1],icolarg[::-1]   

if __name__=="__main__":   
    a = "abc  def  ,hji,,klm  ,  pop, "
    print(a)
    args,icol = splitarg(a)
    for i in icol:
        print(a[i:])
