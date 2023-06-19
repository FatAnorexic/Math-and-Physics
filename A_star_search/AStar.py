class Node():
    def __init__(self,parent=None, position=None):
        self.parent=parent
        self.position=position

        self.g=0
        self.h=0
        self.f=0

    def __eq__(self, other):
        return self.position==other.position

def astar(maze, start, end):
    #create a starting and end position
    start_n=Node(None,start)
    start_n.g=start_n.h=start_n.f=0
    end_n=Node(None, end)
    end_n.g=end_n.h=end_n.f=0

    openL=[]
    closed=[]
    openL.append(start_n)

    while len(openL)>0:
        current=openL[0]
        index=0
        for i, item in enumerate(openL):
            if item.f<current.f:
                current=item
                index=i
        openL.pop(index)
        closed.append(current)

        if current==end_n:
            path=[]
            current_n=current
            while current_n is not None:
                path.append(current_n.position)
                current_n=current_n.parent
            return path[::-1]

        children=[]
        for new_pos in [(0, -1), (0, 1), (-1, 0), (1, 0),
                        (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            node_pos=(current.position[0]+new_pos[0], current.position[1]+new_pos[1])

            if (node_pos[0]>(len(maze)-1) or node_pos[0]<0
                or node_pos[1]>(len(maze[len(maze)-1])-1) or node_pos[1]<0):
                continue
            if maze[node_pos[0]][node_pos[1]] !=0:
                continue
            new_n=Node(current, node_pos)

            children.append(new_n)

        #loop through children
        for child in children:
            for closed_child in closed:
                if child==closed_child:
                    continue
            #create g, f, and heuristic values
            child.g=current.g+1

            child.h=(((child.position[0]-end_n.position[0])*(child.position[0]-end_n.position[0]))+
                     ((child.position[1]-end_n.position[1])*(child.position[1]-end_n.position[1])))

            child.f=child.g+child.h

            for open_node in openL:
                if child==open_node and child.g>open_node.g:
                    continue
            openL.append(child)

def main():
    maze=[[0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    #create two tuples

    start=(0,0)
    end=(7,6)

    path=astar(maze, start, end)
    print(f'Optimal Path:\t{path}')


if __name__=='__main__':
    main()