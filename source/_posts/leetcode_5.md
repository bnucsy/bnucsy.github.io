---
title: LeetCode-5 最长回文子串
categories: [code, LeetCode]
mathjax: true
date: 2022-08-13 09:31:37
---

## [LeetCode-5 最长回文子串](https://leetcode-cn.com/problems/longest-palindromic-substring/)【M】

### 题目描述

给你一个字符串 s，找到 s 中最长的回文子串。

<!-- more -->

- 示例 1：

```
输入：s = "babad"
输出："bab"
解释："aba" 同样是符合题意的答案。
```

- 示例 2：

```
输入：s = "cbbd"
输出："bb"
```

提示：

```
1 <= s.length <= 1000
s 仅由数字和英文字母组成
```

---

### 解答

#### 思路1

对于题目，很容易想到使用动态规划，更新规则应该是`dp[i][j] = dp[i+1][j-1] if s[i]==s[j]`

应该注意的是：数据长度最大值为1000，即可以使用二维数组。

最终可以接受的答案包括两种情况：①关于某字符中心对称  ②关于对称轴对称

采用中心扩展方法：遍历所有可能的中心位置，分对中心对称和轴对称两种情况遍历扩展即可

- 时间复杂度：$O(n^2)$
- 空间复杂度：$O(1)$

#### 代码

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        l,r = 0,len(s)-1
        maxstr = s[0]
        for i in range(len(s)):
            p,v = i,i+1
            while p>=l and v<=r and s[p] == s[v]:
                if v-p+1>len(maxstr):
                    maxstr = s[p:v+1]
                p -= 1
                v += 1
            p = v = i
            while p>=l and v<=r and s[p] == s[v]:
                if v-p+1>len(maxstr):
                    maxstr = s[p:v+1]
                p -= 1
                v += 1
        return maxstr
```

#### 提交结果

时间——$60.1\%$

空间——$98.9\%$

---

#### 思路2

[Manacher 算法](https://www.jianshu.com/p/116aa58b7d81)

好难

估计搞懂了也立刻忘记

- 时间复杂度：$O(n)$
- 空间复杂度：$O(n)$

---





