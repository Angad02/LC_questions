# Top N Buzzwords

class Solution:
    def topNBuzzWords(self, numToys, topToys, toys, numQuotes, quotes):
        # Base case: if toys list is empty or quotes list is empty, the output list should also be []
        if not toys or not quotes:
            return []

        # Base case: if topToys are 0, that means we don't have to return any buzzwords
        if topToys == 0:
            return []

        # Dict to store toy buzzword as key and value as tuple in form of (total_freq, total_quotes)
        # So we will have dict like: {'elmo': (4, 3), 'elsa': (4, 2)}
        # Here, 'elmo': (4, 3) means word 'elmo' occurs total 4 times throughout all the quotes and it comes in 3 different quotes.
        #
        # We will first create entry for all toys with (total_freq, total_quotes) = (0, 0)
        toy_freq_quote = dict()
        for t in toys:
            toy_freq_quote[t] = (0, 0)

        # Iterate through all the quotes
        for q in quotes:
            # We need this updated_quote_count dict so that we don't increment the quote count for a buzzword more than once,
            # in case if it occurs multiple times in a single quote
            updated_quote_count = {toy: False for toy in toys}

            # Convert all the words to lowercase and split them.
            # Go through all the words of a quote and:
            #   - Remove all the extra characters from the words except "a-z". Basically we replace them with '' using regex.
            for w in q.lower().split():
                # We don't need to include A-Z in the regex, because we have already turned everything to lowercase before splitting.
                w = re.sub('[^a-z]', '', w)

                # Check if the current word is a toy/buzzword
                if toy_freq_quote.get(w):
                    # Get current frequency and quote counts
                    curr_freq, curr_quote = toy_freq_quote[w][0], toy_freq_quote[w][1]
                    # If the current quote count is not already incremented for word w, do it and mark it in updated_quote_count
                    if not updated_quote_count[w]:
                        curr_quote += 1
                        updated_quote_count[w] = True

                    # Update freq and quote_count values
                    toy_freq_quote[w] = (curr_freq+1, curr_quote)

        # Initially toy_freq_quote was created for all the given toys.
        # It is possible that some of those just don't appear in quotes.
        # Remove such toys from the toy_freq_quote whose frequency is 0
        for t in toys:
            if toy_freq_quote[t][0] == 0:
                del toy_freq_quote[t]

        # Now, we have the dict ready with all the buzzwords from toys list that come in quotes,
        # along with their total frequency and total quote count.
        #
        # First thing to check is if topToys (i.e. number of top toys/buzzwords to return) is > total numToys (i.e. total buzzwords)
        #   - If it is, then as per the given requirement, we just need to return the list of present buzzwords in quotes.
        #
        # If we return here, then all the computation stops and we are done.
        if topToys > numToys:
            return [toy for toy in toy_freq_quote]

        # Declare a list which we can use as heap.
        buzzword_heap = []

        # Go through all buzzwords from toy_freq_quote dict and take their (total_freq, total_quote_count).
        # Add it to the buzzword_heap as (-1*total_freq, -1*total_quote_count, buzzword)
        #
        # - Since we want to order by maximum total_freq and after that maximum total_quote_count, we will multiply them with -1 when
        #   pushing into the heap.
        # - We do that because in Python heapq.heapify(list), the heap created is min-heap.
        # - If we don't multiply those numbers by -1, then we will output with minimum total_freq and total_quotes.
        # - Also, we keep the ordering like (total_freq, total_quote_count, buzzword)
        #   because we first have to get the buzzword with max frequency, after that with max quote_count, and in the end
        #   in alphabetical order of buzzwords themselves.
        for toy in toy_freq_quote:
            total_freq, total_quote_count = toy_freq_quote[toy][0], toy_freq_quote[toy][1]
            buzzword_heap.append((-1*total_freq, -1*total_quote_count, toy))

        heapq.heapify(buzzword_heap)

        # Final result list
        top_buzzwords = []

        # Now we just do heappop equal to the total number of top buzzwords we have to return.
        # For every heappop, we will get back the tuple we have pushed:
        #   (total_freq, total_quote_count, buzzword)
        #
        # Since, in final output, all we need is buzzword, we just take the last element of the tuple, and append it to final list.
        for i in range(topToys):
            toy = heapq.heappop(buzzword_heap)[2]
            top_buzzwords.append(toy)

            # Check if there are any buzzwords left in the buzzword_heap or not.
            # There can be a case where we are asked to return top 5 buzzwords, but among all the quotes, only 3 buzzwords are present
            # In such case, we will return whatever buzzwords are present as per the sorting requirements given
            # This should be conveyed and discussed with interviewer
            if not buzzword_heap:
                break

        return top_buzzwords


# Zombie in Matrix aka Min hours to send file to all available servers 
# Given a 2D grid, each cell is either a zombie 1 or a human 0. Zombies can turn adjacent (up/down/left/right) human beings into zombies every hour. Find out how many hours does it take to infect all humans?
class Solution:
    def minHour(self, rows, columns, grid):
        if not rows or not columns:
            return 0
        
        q = [[i,j] for i in range(rows) for j in range(columns) if grid[i][j]==1]
        directions = [[1,0],[-1,0],[0,1],[0,-1]]
        time = 0
        
        while True:
            new = []
            for [i,j] in q:
                for d in directions:
                    ni, nj = i + d[0], j + d[1]
                    if 0 <= ni < rows and 0 <= nj < columns and grid[ni][nj] == 0:
                        grid[ni][nj] = 1
                        new.append([ni,nj])
            q = new
            if not q:
                break
            time += 1
            
        return time


# Critical Routers

from collections import defaultdict
def findcriticalnodes(n, edges):
    g = defaultdict(list)
    for conn in edges:
        g[conn[0]].append(conn[1])
        g[conn[1]].append(conn[0])
    visited = [0]* n
    isarticulationpoints = [0]*n
    order = [0]*n
    low = [0]*n
    seq = 0

    def dfs(u, p):
        nonlocal seq
        visited[u] = 1
        order[u] = low[u] = seq
        seq = seq + 1
        children = 0 
        for to in g[u]:
            if to == p:
                continue
            if visited[to]:
                low[u] = min(low[u], low[to])
            else:
                dfs(to, u)
                low[u] = min(low[u], low[to])
                if order[u] <= low[to] and p!= -1:
                    isarticulationpoints[u] = 1
                children += 1
        
        if p == -1 and children > 1:
            isarticulationpoints[u] = 1
    
    dfs(0, -1)
    ans = []
    for i in range(len(isarticulationpoints)):
        if isarticulationpoints[i]:
            ans.append(i)
    return ans
    
if __name__ == "__main__":
    a = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 5], [5, 6], [3, 4]]
    print(findcriticalnodes(7, a)

# Search Suggestions System/ Product Suggestions

from collections import defaultdict

class TrieNode:
    def __init__(self):
        self.children = defaultdict(TrieNode)
        self.terminating = False
        
class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, word):
        node = self.root
        for ch in word:          
            node = node.children[ch]
        node.terminating = True
         
    def _iterate_children(self, node, word):
        for k, v in node.children.items():
            v = node.children[k]
            if v.terminating:
                yield word+k
        
            yield from self._iterate_children(v, word+k)
            
    def search(self, word):
        node = self.root
        for ch in word:
            if not node.children.get(ch):
                return None
            node = node.children[ch]
        if node.terminating:
            yield word 
            
        for word in self._iterate_children(node, word):
            yield word

class Solution:
    def suggestedProducts(self, products: List[str], searchWord: str) -> List[List[str]]:
        t = Trie()
        for p in sorted(products):
            t.insert(p)
        
        res = []
        for i in range(1, len(searchWord)+1, 1):
            sug = []
            for suggestion in t.search(searchWord[:i]):
                sug.append(suggestion)
                if len(sug) >= 3:
                    break
            res.append(sug)
        return res

# Number of Clusters
class Solution:
    def numIslands(self, grid: List[List[str]]) -> int:
        """
        :type grid: List[List[str]]
        :rtype: int
        """
        if not grid: return 0
        r, c = len(grid), len(grid[0])
        visited = [[False for _ in range(c)] for _ in range(r)]

        def dfs(i, j):
            if i < 0 or i >= r or j < 0 or j >= c or grid[i][j] == '0' or visited[i][j]:
                return
            visited[i][j] = True
            dfs(i + 1, j)
            dfs(i - 1, j)
            dfs(i, j + 1)
            dfs(i, j - 1)

        count = 0
        for i in range(r):
            for j in range(c):
                if not visited[i][j] and grid[i][j] == '1':
                    dfs(i, j)
                    count += 1
        return count

# Reorder Data in Log Files
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        let_logs = []
        dig_logs = []
        for log in logs:
            log_key, log_val = log.split(" ", 1)
            if log_val[0].isalpha():
                let_logs.append(log_val + " " + log_key)
            else:
                dig_logs.append(log)
        let_logs.sort()
        for i, let_log in enumerate(let_logs):
            x = let_log.split()
            let_logs[i] = x[-1] + " " + " ".join(x[:-1])

        return let_logs + dig_logs

# Optimal Utilization
def findPairs(a, b, target):
	a.sort(key=lambda x: x[1])
	b.sort(key=lambda x: x[1])
	l, r = 0, len(b) - 1
	ans = []
	curDiff = float('inf')
	while l < len(a) and r >= 0:
		id1, i = a[l]
		id2, j = b[r]
		if (target - i - j == curDiff):
			ans.append([id1, id2])
		elif (i + j <= target and target - i - j < curDiff):
			ans.clear()
			ans.append([id1, id2])
			curDiff = target - i - j
		if (target > i + j):
			l += 1
		else:
			if target == i + j:
				tmp_l = l
				while a[tmp_l][1] + b[r][1] == target:
					tmp_l += 1
					if tmp_l == len(a):
						break
					if  a[tmp_l][1] + b[r][1] == target:
						ans.append([a[tmp_l][0], b[r][0]])
			r -= 1
	
	ans.sort(key = lambda x: x[1])
	ans.sort(key = lambda x: x[0])
	return ans

# Min Cost to Connect Ropes
from heapq import heappop, heappush, heapify
def minCost(ropes: List[int]) -> int:
  if not ropes: return 0
  if len(ropes) == 1: return ropes[0]
  heapify(ropes)
  cost = 0
  while len(ropes) > 1:
    a, b = heappop(ropes), heappop(ropes)
    cost += a+b
    if ropes:
      heappush(ropes, a+b)
  return cost

  # Treasure Island (Treasure Island / Min Distance to Remove the Obstacle)
def solution(m):
    if len(m) == 0 or len(m[0]) == 0:
        return -1  # impossible

    matrix = [row[:] for row in m]
    nrow, ncol = len(matrix), len(matrix[0])

    q = deque([((0, 0), 0)])  # ((x, y), step)
    matrix[0][0] = "D"
    while q:
        (x, y), step = q.popleft()

        for dx, dy in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
            if 0 <= x+dx < nrow and 0 <= y+dy < ncol:
                if matrix[x+dx][y+dy] == "X":
                    return step+1
                elif matrix[x+dx][y+dy] == "O":
                    # mark visited
                    matrix[x + dx][y + dy] = "D"
                    q.append(((x+dx, y+dy), step+1))

    return -1

# Treasure Island II
def find_treasure(t_map, row, col, curr_steps, min_steps):
    if row >= len(t_map) or row < 0 or col >= len(t_map[0]) or col < 0 or t_map[row][col] == 'D' or t_map[row][col] == '#':
        return None, min_steps

    if t_map[row][col] == 'X':
        curr_steps += 1
        if min_steps > curr_steps:
            min_steps = min(curr_steps, min_steps)

        return None, min_steps

    else:
        tmp = t_map[row][col]
        t_map[row][col] = '#'
        curr_steps += 1
        left = find_treasure(t_map, row, col-1, curr_steps, min_steps)
        right = find_treasure(t_map, row, col+1, curr_steps, min_steps)
        up = find_treasure(t_map, row-1, col, curr_steps, min_steps)
        down = find_treasure(t_map, row+1, col, curr_steps, min_steps)

        t_map[row][col] = tmp

        return curr_steps, min(left[1], right[1], up[1], down[1])


def main(tr_mp):
    main_min_steps = float('inf')
    start = None
    for row in range(len(treasure_map)):
        for col in range(len(treasure_map[0])):
            # min_res = [0, 0]
            if treasure_map[row][col] == 'S':
                min_res = find_treasure(tr_mp, row, col, -1, main_min_steps)

            if min_res[1] < main_min_steps:
                start = row, col
                main_min_steps = min_res[1]

    return main_min_steps, start


if __name__ == '__main__':
    treasure_map = [['S', '1', '1', 'X'],
                    ['D', 'D', 'X', '1'],
                    ['1', 'S', '1', '1'],
                    ['1', 'X', 'X', '1']]
    print(main(treasure_map))
    
######################## JAVA SOLUTION ##############################################
# public class Main {
#     private static final int[][] dirs = new int[][]{{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    
#     private static int minDistToTreasure(char[][] island) {
#         if (island == null || island.length == 0 || island[0].length == 0)
#             return -1;
        
#         int minDist = 0;
        
#         Deque<int[]> que = new LinkedList<>();
        
#         for (int i = 0; i < island.length; i++) {
#             for (int j = 0; j < island[0].length; j++) {
#                 if (island[i][j] == 'X') {
#                     que.addLast(new int[]{i, j});
#                     island[i][j] = 'V';
#                 }
#             }
#         }
        
#         while (!que.isEmpty()) {
#             int qSize = que.size();
            
#             while (qSize > 0) {
#                 int[] item = que.removeFirst();
                
#                 for (int[] dir: dirs) {
#                     int row = item[0] + dir[0];
#                     int col = item[1] + dir[1];
                    
#                     if (row >= 0 && row < island.length && col >= 0 && col < island[0].length &&
#                        island[row][col] != 'D' && island[row][col] != 'V') {
#                         if (island[row][col] == 'S')
#                             return minDist + 1;
#                         else
#                             island[row][col] = 'V';
                        
#                         que.addLast(new int[]{row, col});
#                     }
#                 }
                
#                 qSize--;
#             }
#             minDist++;
#         }
        
#         return -1;
#     }
    
#     public static void main(String[] args) {
#         char[][] island = new char[][] {
#             {'S', 'O', 'O', 'S', 'S'},
#             {'D', 'O', 'D', 'O', 'D'},
#             {'O', 'O', 'O', 'O', 'X'},
#             {'X', 'D', 'D', 'O', 'O'},
#             {'X', 'D', 'D', 'D', 'O'}};
#         System.out.println(minDistToTreasure(island));
#     }
# }

# Find Pair With Given Sum
def twoNumberSum(array, targetSum):
	for i in range(len(array)-1):
		first_number = array[i]
		for j in range(i+1, len(array)):
			if first_number + array[j] == targetSum:
				return sorted([first_number, array[j]])
	return []


# Copy List with Random Pointer
"""
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
"""
class Solution:
    def copyRandomList(self, head: 'Node') -> 'Node':
        if head == None:
            return None
        
        table = {}
        tmp = head
        while tmp:#create node and build hash table
            node = Node(tmp.val, None, None)
            table[tmp] = node
            tmp = tmp.next
        
        tmp = head
        while tmp:
            if tmp.next in table:#assign next
                table[tmp].next = table[tmp.next]
            else:
                table[tmp].next = None
            if tmp.random in table:#assign random
                table[tmp].random = table[tmp.random]
            else:
                table[tmp].random = None
            #print(table[tmp].val)
            #print(tmp.val)
            tmp = tmp.next
        
        return table[head]
        
# Merge Two Sorted Lists
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if not l1:
            return l2
        if not l2:
            return l1
        
        dummyNode = ListNode(0)
        pre = dummyNode
        
        while l1 and l2:
            if l1.val <= l2.val:
                pre.next = l1
                l1= l1.next
            else:
                pre.next = l2
                l2 = l2.next
            pre=pre.next
        #if still remaining in either l1 or l2
        if not l1:
            pre.next = l2
        elif not l2:
            pre.next = l1
        # pre.next = l1 or l2
        
        return dummyNode.next

# Subtree of Another Tree

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def check(self, s, t):
        if s is None or t is None:
            return s is None and t is None
        if s.val != t.val:
            return False
        return self.check(s.left, t.left) and self.check(s.right, t.right)

    def isSubtree(self, s: TreeNode, t: TreeNode) -> bool:
        if s is None or t is None:
            return s is None and t is None
        if self.check(s, t):
            return True
        return self.isSubtree(s.left, t) or self.isSubtree(s.right, t)


# Search a 2D Matrix II
class Solution:
    def searchMatrix(self, matrix, target):
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        nrow = len(matrix)
        if not nrow:
           return False
        ncol = len(matrix[0])
        row, col = 0, ncol-1
        while row < nrow and col >= 0:
            if matrix[row][col] == target:
                return True
            elif matrix[row][col] > target:
                col -= 1
            else:
                row += 1
        return False


# Critical Connections

from collections import defaultdict


class Graph:
    def __init__(self, vertices):
        self.V = vertices + 1
        self.adj_list = defaultdict(list)
        self.parent = [-1] * self.V
        self.low = [float('inf')] * self.V
        self.disc = [float('inf')] * self.V
        self.visited = [False] * self.V
        self.time = 0
        self.bridges = []

    def add_edge(self, v, u):
        self.adj_list[u].append(v)
        self.adj_list[v].append(u)

    def find_bridges(self, u):
        # to keep track of which vertices are visited
        self.visited[u] = True
        # to check the lowest vertex that this node can reach through its children
        self.low[u] = self.time
    # the time that this vertex was discovered - useful to identify when a child vertex can reach its parents ancestors
        self.disc[u] = self.time
        # a way to record the when a vertex was first encountered
        self.time += 1

        # loop over each child vertex of u
        for v in self.adj_list[u]:

            # if the child vertex v has not been visited then enter here
            if not self.visited[v]:
                # mark the parent of v as u as it was first discovered by u
                self.parent[v] = u
                # depth first search over its children
                self.find_bridges(v)

                # once we complete the dfs and come back here
                # we update the lowest vertex that we can reach from any child of u
                self.low[u] = min(self.low[u], self.low[v])

                # if the lowest vertex of child v is greater [which means below] then when u was discovered,
                # then it is assumed that v does not have a way back to u
                # and therefore v and u are not in a cycle and hence a bridge
                if self.low[v] > self.disc[u]:
                    self.bridges.append([u, v])

# if v is already visited then we check if v is the parent of u, if yes then that means no need to update the lowest
                # vertex for u and v is above it.
                # else it means v is an ancestor of u and there is a back edge between u or subtree rooted at u back
                # to v and we need to find the earliest time that was used to visit u using the logic mentioend below.
            elif v != self.parent[u]:
                self.low[u] = min(self.low[u], self.disc[v])


if __name__ == '__main__':
    g = Graph(6)
    # edges = [[1, 2], [1, 3], [3, 4], [1, 4], [4, 5]]
    # edges = [[1, 2], [1, 3], [2, 3], [3, 4], [3, 6], [4, 5], [6, 7], [6, 9], [7, 8], [8, 9]]
    edges = [[1, 2], [1, 3], [2, 3], [2, 4], [2, 5], [4, 6], [5, 6]]
    for i in edges:
        g.add_edge(i[0], i[1])

    for j in range(1, g.V):
        print(j)
        if not g.visited[j]:
            g.find_bridges(j)

    print(g.adj_list)
    print(g.bridges)


# Favorite Genres
def favGenres(userSongs, songGenres):
    output={}
    for i in userSongs:
        list=userSongs[i]
        count=collections.defaultdict(int)
        for j in list:
            for k,v in songGenres.items():
                if j in v:
                    count[k]+=1

        print(count)
        output[i]=[key for key,val in count.items() if val ==max(count.values())]
                
    
    return output


#Two Sum - Unique Pairs
def uniqueTwoSum(nums, target):
  ans, comp = set(), set()
  for n in nums:
    c = target-n
    if c in comp:
      res = (n, c) if n > c else (c, n)
      if res not in ans:
        ans.add(res)
    comp.add(n)
  return len(ans)


# Spiral Matrix II
class Solution:
    def generateMatrix(self, n: int) -> List[List[int]]:
        if n == 1:
            return [[1]]
        
        p = n
        return_matrix = [[i for i in range(p)] for i in range(n)]
        right = n
        down = n - 1
        left = n - 1
        up = n - 2
        col = 0
        row = 0
        k = 1
        while right>0 or down>0 or left>0 or up>0:
            for i in range(right):
                return_matrix[row][col] = k
                col+=1
                k+=1
            col -= 1
            for i in range(down):
                row+=1
                return_matrix[row][col] = k
                k += 1 

            for i in range(left):
                col -= 1
                return_matrix[row][col] = k
                k += 1

            for i in range(up):
                row -= 1
                return_matrix[row][col] = k
                k += 1
            col+=1

            right -= 2 
            down -= 2
            left -= 2
            up -= 2
        return return_matrix


# Count substrings with exactly K distinct chars
import collections
def subStringsWithKDistinctCharacters(s, k):
    s = list(s)
    
    def atMost(k):
        count = collections.defaultdict(int)
        left = 0
        ans = 0
        for right, x in enumerate(s):
            count[x] += 1
            while len(count) > k:
                count[s[left]] -= 1
                if count[s[left]] == 0:
                    del count[s[left]]
                left += 1
            ans += right - left + 1
        return ans
    return atMost(k) - atMost(k-1)

s = "aabab"
k = 3
print(subStringsWithKDistinctCharacters(s, k))

# Max Of Min Altitudes
# class Solution:
def sol(nums):

    N = len(nums)
    M = len(nums[0])

    nums[0][0] = 1e9
    nums[N - 1][M - 1] = 1e9

    dp = [[1e9] * M for i in range(N)]

    for j in range(1, M):
        dp[0][j] = min(dp[0][j - 1], nums[0][j])
    for i in range(1, N):
        dp[i][0] = min(dp[i - 1][0], nums[i][0])

    for i in range(1, N):
        for j in range(1, M):
            cur = max(dp[i - 1][j], dp[i][j - 1])
            dp[i][j] = min(cur, nums[i][j])
    #print(dp)

    print("ans: " + str(dp[N - 1][M - 1]))
    
nums = [[1, 2, 3], [4, 5, 1]]
sol(nums)


#  Longest Palindromic Substring
class Solution:
    def longestPalindrome(self, s: str) -> str:
        if len(s) == 0:
            return s
        for l in range(len(s),0,-1):
            for i in range(0,len(s)-l+1):
                if s[i:i+l] == s[i:i+l][::-1]:
                    return(s[i:i+l])


# Substrings of size K with K distinct chars
def substringk(s, k):
    if not s or k == 0:
        return []
    
    letter, res = {}, set()
    start = 0
    for i in range(len(s)):
        if s[i] in letter and letter[s[i]] >= start:
            start = letter[s[i]]+1
        letter[s[i]] = i
        if i-start+1 == k:
            res.add(s[start:i+1])
            start += 1
    return list(res)

# Most Common Word
class Solution:
    def mostCommonWord(self, paragraph: str, banned: List[str]) -> str:
        banset = set(banned)
        for c in "!?',;.":
            paragraph = paragraph.replace(c, " ")
        count = collections.Counter(
            word for word in paragraph.lower().split())

        ans, best = '', 0
        for word in count:
            if count[word] > best and word not in banset:
                ans, best = word, count[word]

        return ans

# K Closest Points to Origin

class Solution:
    def kClosest(self, points: List[List[int]], K: int) -> List[List[int]]:
         points.sort(key = lambda P: P[0]**2 + P[1]**2)
         return points[:K]

# Generate Parentheses
def generateParenthesis(self, N):
        if N == 0: return ['']
        ans = []
        for c in xrange(N):
            for left in self.generateParenthesis(c):
                for right in self.generateParenthesis(N-1-c):
                    ans.append('({}){}'.format(left, right))
        return ans

# Min Cost to Connect All Nodes
def compute_min_cost(num_nodes, base_mst, poss_mst):
    uf = {}

    # create union find for the initial edges given 
    def find(edge):
        uf.setdefault(edge, edge)
        if uf[edge] != edge:
            uf[edge] = find(uf[edge])
        return uf[edge]

    def union(edge1, edge2):
        uf[find(edge1)] = find(edge2)

    for e1, e2 in base_mst:
        if find(e1) != find(e2):
            union(e1, e2)

    # sort the new edges by cost
    # if an edge is not part of the minimum spanning tree, then include it, else continue
    cost_ret = 0
    for c1, c2, cost in sorted(poss_mst, key=lambda x : x[2]):
        if find(c1) != find(c2):
            union(c1, c2)
            cost_ret += cost

    if len({find(c) for c in uf}) == 1 and len(uf) == num_nodes:
        return cost_ret
    else:
        return -1


if __name__ == '__main__':
    n = 6
    edges = [[1, 4], [4, 5], [2, 3]]
    new_edges = [[1, 2, 5], [1, 3, 10], [1, 6, 2], [5, 6, 5]]
    print(compute_min_cost(n, edges, new_edges))


# Min Cost to Repair Edges

def minimumCost(N: int, roads: List[List[int]], repairs: List[List[int]]) -> int:
  cost_map = {}
  for c1, c2, c in repairs:
    cost_map[(c1, c2)] = c
    
  for edge in roads:
    c1, c2 = edge
    if (c1, c2) not in cost_map:
      cost_map[(c1, c2)] = 0
    
  connections = []
  for key in cost_map:
    c1, c2 = key
    connections.append([c1, c2, cost_map[key]])
    
    
  if len(connections) < N - 1:
    return -1
  if N is 1:
    return 0

  connections = sorted(connections, key=lambda x: x[2])
  total_cost = 0

  parent = {}

  def findParent(city):
    parent.setdefault(city, city)
    if parent[city] == city:
      return city
    else:
      return findParent(parent[city])

  def mergeSets(city1, city2):
    parent1, parent2 = findParent(city1), findParent(city2)
    if parent1 != parent2:
      parent[parent1] = parent2
      return cities - 1, total_cost + cost
    return cities, total_cost

  cities = N - 1
  for city1, city2, cost in connections:
    cities, total_cost = mergeSets(city1, city2)
          
  return total_cost if cities == 0 else -1

print(minimumCost(N = 5, roads = [[1, 2], [2, 3], [3, 4], [4, 5], [1, 5]], repairs = [[1, 2, 12], [3, 4, 30], [1, 5, 8]]))
print(minimumCost(6, [[1, 2], [2, 3], [4, 5], [3, 5], [1, 6], [2, 4]], [[1, 6, 410], [2, 4, 800]]))
print(minimumCost(N = 6, roads = [[1, 2], [2, 3], [4, 5], [5, 6], [1, 5], [2, 4], [3, 4]], repairs = [[1, 5, 110], [2, 4, 84], [3, 4, 79]]))


# Prison Cells After N Days
class Solution:
    def prisonAfterNDays(self, cells: List[int], N: int) -> List[int]:
        while N>0:
            cells2 = []
            cells2.append(0)
            for i in range(1, 7):
                val = 1 if cells[i-1]==cells[i+1] else 0
                cells2.append(val);
            cells2.append(0)
            cells = cells2
            N = (N - 1)%14 # needed or else timeout
            # N = N - 1 # brute force
        return cells;
        

#  Partition Labels

class Solution:
    def partitionLabels(self, S: str) -> List[int]:
        last = {c: i for i, c in enumerate(S)}
        j = anchor = 0
        ans = []
        for i, c in enumerate(S):
            j = max(j, last[c])
            if i == j:
                ans.append(i - anchor + 1)
                anchor = i + 1
            
        return ans

# Subtree with Maximum Average

class TreeNode:
    def __init__(self, value):
        self.val = value
        self.children = []
        

class Solution: 
    def MaxAverageSubtree(self, root):
        if not root or not root.children:
            return None
        
        self.res = [float('-inf'), 0]
        # self.res[0]: average; self.res[1]: number of nodes
        self.dfs(root)
        return self.res[1]
    
    def dfs(self, root):
        if not root.children:
            return [root.val, 1]
        
        temp_sum, temp_num = root.val, 1
        for child in root.children:
            child_sum, child_num = self.dfs(child)
            temp_sum += child_sum
            temp_num += child_num
            
        if temp_sum/temp_num > self.res[0]:
            self.res = [temp_sum/temp_num, root.val]
        
        return [temp_sum, temp_num]