## Getting the TF workbench ready
The first thing we need to do in our jupyter notebook is to
1. load the TF program
2. load the TF database


```python
# First, I have to laod different modules that I use for analyzing the data and for plotting:
import sys, os, collections
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt; plt.rcdefaults()
from matplotlib.pyplot import figure
from collections import Counter
```


```python
# First we load the TF program
from tf.fabric import Fabric
from tf.app import use
```


```python
%%time
# Now we load a few databases
#BHSaLatest = use('etcbc/bhsa:latest')
#BHSaC = use('etcbc/bhsa', version='c', mod='CenterBLC/BHSaddons/tf')
#BHSa2017 = use('etcbc/bhsa', version='2017', mod='CenterBLC/BHSaddons/tf')

BHS = use('etcbc/bhsa', version="2021", mod='CenterBLC/BHSaddons/tf', hoist=globals())
bhsF, bhsL, bhsT, bhsS = BHS.api.F, BHS.api.L, BHS.api.T, BHS.api.S

#LXX = use("CenterBLC/LXX", version="1935", hoist=globals())
#lxxF, lxxL, lxxT, lxxS = LXX.api.F, LXX.api.L, LXX.api.T, LXX.api.S
#
#DSS = use('etcbc/dss', hoist=globals())
#dssF, dssL, dssT, dssS = DSS.api.F, DSS.api.L, DSS.api.T, DSS.api.S
#
#GNT = use ("CenterBLC/N1904", version="1.0.0", mod="CenterBLC/N1904/BOLcomplement/tf/", hoist=globals())
#gntF, gntL, gntT, gntS = GNT.api.F, GNT.api.L, GNT.api.T, GNT.api.S
```


**Locating corpus resources ...**



<b title="local commit">app:</b> <span title="#gb112c161cfd21eae403d51a2733740d8743460e7 offline under /Users/oliverglanz/text-fabric-data/github">~/text-fabric-data/github/etcbc/bhsa/app</span>



<b title="local commit">data:</b> <span title="#gb112c161cfd21eae403d51a2733740d8743460e7 offline under /Users/oliverglanz/text-fabric-data/github">~/text-fabric-data/github/etcbc/bhsa/tf/2021</span>



<b title="local commit">data:</b> <span title="#15b8f36062526e055b782e643938e44eeb7cd6cd offline under /Users/oliverglanz/text-fabric-data/github">~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021</span>




            <b>TF:</b> <a target="_blank" href="https://annotation.github.io/text-fabric/tf/cheatsheet.html" title="text-fabric api">TF API 13.1.0</a>, <a target="_blank" href="https://github.com/etcbc/bhsa/blob/master/app" title="etcbc/bhsa app">etcbc/bhsa/app  v3</a>, <a target="_blank" href="https://annotation.github.io/text-fabric/tf/about/searchusage.html" title="Search Templates Introduction and Reference">Search Reference</a><br>
            <b>Data:</b> <a target="_blank" href="https://etcbc.github.io/bhsa/" title="provenance of BHSA = Biblia Hebraica Stuttgartensia Amstelodamensis">etcbc - bhsa 2021</a>, <a target="_blank" href="https://annotation.github.io/text-fabric/tf/writing/hebrew.html" title="How TF features represent text">Character table</a>, <a target="_blank" href="https://etcbc.github.io/bhsa/features/0_home" title="etcbc - bhsa feature documentation">Feature docs</a><br>
            <details class="nodeinfo"><summary><b>Node types</b></summary>
<table class="nodeinfo">
    <tr>
        <th>Name</th>
        <th># of nodes</th>
        <th># slots / node</th>
        <th>% coverage</th>
    </tr>

<tr>
    <th>book</th>
    <td>39</td>
    <td>10938.21</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>chapter</th>
    <td>929</td>
    <td>459.19</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>lex</th>
    <td>9230</td>
    <td>46.22</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>verse</th>
    <td>23213</td>
    <td>18.38</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>half_verse</th>
    <td>45179</td>
    <td>9.44</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>sentence</th>
    <td>63717</td>
    <td>6.70</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>sentence_atom</th>
    <td>64514</td>
    <td>6.61</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>clause</th>
    <td>88131</td>
    <td>4.84</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>clause_atom</th>
    <td>90704</td>
    <td>4.70</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>phrase</th>
    <td>253203</td>
    <td>1.68</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>phrase_atom</th>
    <td>267532</td>
    <td>1.59</td>
    <td><b>100</b></td>
</tr>

<tr>
    <th>subphrase</th>
    <td>113850</td>
    <td>1.42</td>
    <td>38</td>
</tr>

<tr>
    <th><i>word</i></th>
    <td>426590</td>
    <td>1.00</td>
    <td><b>100</b></td>
</tr>
</table></details>
            <b>Sets:</b> no custom sets<br>
            <b>Features:</b><br>
<details><summary><b><github>CenterBLC/BHSaddons/tf</b></summary>
    <div class="fcorpus">

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/BHS5_CA.tf">BHS5_CA</a>
</div>
<div class="fmono">str</div>

<span> ✅ each word in its consonantal form only and with not disambiguation (BR>, >LHJM)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/BHS5_text.tf">BHS5_text</a>
</div>
<div class="fmono">str</div>

<span> ✅ each word in its consonantal form only and with not disambiguation (BR>, >LHJM)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/bol_bhsa_word_order.tf">bol_bhsa_word_order</a>
</div>
<div class="fmono">str</div>

<span> ✅ the sequence number of each word in the BHSa (1,2,3,...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/bol_dict_EN.tf">bol_dict_EN</a>
</div>
<div class="fmono">str</div>

<span> ✅ English glosses of the BibleOL Hebrew/Aramaic dictionary</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/bol_dict_ES.tf">bol_dict_ES</a>
</div>
<div class="fmono">str</div>

<span> ✅ Spanish glosses of the BibleOL Hebrew/Aramaic dictionary</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/bol_dict_HebArm.tf">bol_dict_HebArm</a>
</div>
<div class="fmono">str</div>

<span> ✅ BibleOL Hebrew/Aramaic dictionary with homograph disambiguation (ברא I. ברא II)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/bol_dict_abc.tf">bol_dict_abc</a>
</div>
<div class="fmono">str</div>

<span> ✅ The dictionary sequence number of each word (1,2,3,...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/bol_dict_vc.tf">bol_dict_vc</a>
</div>
<div class="fmono">str</div>

<span> ✅ BibleOL verbal classes (i-nun, ii-waw, ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/g_word_noaccent.tf">g_word_noaccent</a>
</div>
<div class="fmono">str</div>

<span> ✅ each text-realized word with consonants and vowels but no Masoretic signs (B.@R@>, >:ELOHIJM)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/interlinear.tf">interlinear</a>
</div>
<div class="fmono">str</div>

<span> ✅ BibleOL verbal classes (i-nun, ii-waw, ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/lex_cons.tf">lex_cons</a>
</div>
<div class="fmono">str</div>

<span> ✅ each word in its consonantal form only and with not disambiguation (BR>, >LHJM)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/macula_BH_lxx_lex_1.tf">macula_BH_lxx_lex_1</a>
</div>
<div class="fmono">str</div>

<span> ✅ macula BH-LXX_lex1</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/macula_BH_lxx_lex_2.tf">macula_BH_lxx_lex_2</a>
</div>
<div class="fmono">str</div>

<span> ✅ macula BH-LXX_lex2</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/macula_BH_lxx_lex_3.tf">macula_BH_lxx_lex_3</a>
</div>
<div class="fmono">str</div>

<span> ✅ macula BH-LXX_lex3</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/macula_BH_lxx_lex_4.tf">macula_BH_lxx_lex_4</a>
</div>
<div class="fmono">str</div>

<span> ✅ macula BH-LXX_lex4</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/macula_BH_lxx_text.tf">macula_BH_lxx_text</a>
</div>
<div class="fmono">str</div>

<span> ✅ macula BH-LXX_text</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/rahlfs_lex_1.tf">rahlfs_lex_1</a>
</div>
<div class="fmono">str</div>

<span> ✅ parallel Rahlfs lex1</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/rahlfs_monad_num.tf">rahlfs_monad_num</a>
</div>
<div class="fmono">str</div>

<span> ✅ parallel Rahlfs monad num</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/rahlfs_reference.tf">rahlfs_reference</a>
</div>
<div class="fmono">str</div>

<span> ✅ parallel Rahlfs book, chapter, verse reference</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/rahlfs_text.tf">rahlfs_text</a>
</div>
<div class="fmono">str</div>

<span> ✅ parallel Rahlfs text</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/strong.tf">strong</a>
</div>
<div class="fmono">str</div>

<span> ✅ Strong's numbers</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/tov_BH_text.tf">tov_BH_text</a>
</div>
<div class="fmono">str</div>

<span> ✅ parallel Tov's MT</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/tov_LXX_lex.tf">tov_LXX_lex</a>
</div>
<div class="fmono">str</div>

<span> ✅ Tov's GT lex</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://github.com/CenterBLC/BHSaddons/tree/master/tf" title="~/text-fabric-data/github/CenterBLC/BHSaddons/tf/2021/tov_LXX_text.tf">tov_LXX_text</a>
</div>
<div class="fmono">str</div>

<span> ✅ Tov's GT</span>

</div>

    </div>
</details>

<details><summary><b>BHSA = Biblia Hebraica Stuttgartensia Amstelodamensis</b></summary>
    <div class="fcorpus">

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/book" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/book.tf">book</a>
</div>
<div class="fmono">str</div>

<span> ✅ book name in Latin (Genesis; Numeri; Reges1; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/book@ll" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/book@am.tf">book@ll</a>
</div>
<div class="fmono">str</div>

<span> ✅ book name in amharic (ኣማርኛ)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/chapter" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/chapter.tf">chapter</a>
</div>
<div class="fmono">int</div>

<span> ✅ chapter number (1; 2; 3; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/code" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/code.tf">code</a>
</div>
<div class="fmono">int</div>

<span> ✅ identifier of a clause atom relationship (0; 74; 367; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/det" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/det.tf">det</a>
</div>
<div class="fmono">str</div>

<span> ✅ determinedness of phrase(atom) (det; und; NA.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/domain" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/domain.tf">domain</a>
</div>
<div class="fmono">str</div>

<span> ✅ text type of clause (? (Unknown); N (narrative); D (discursive); Q (Quotation).)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/freq_lex" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/freq_lex.tf">freq_lex</a>
</div>
<div class="fmono">int</div>

<span> ✅ frequency of lexemes</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/function" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/function.tf">function</a>
</div>
<div class="fmono">str</div>

<span> ✅ syntactic function of phrase (Cmpl; Objc; Pred; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/g_cons" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/g_cons.tf">g_cons</a>
</div>
<div class="fmono">str</div>

<span> ✅ word consonantal-transliterated (B R>CJT BR> >LHJM ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/g_cons_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/g_cons_utf8.tf">g_cons_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ word consonantal-Hebrew (ב ראשׁית ברא אלהים)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/g_lex" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/g_lex.tf">g_lex</a>
</div>
<div class="fmono">str</div>

<span> ✅ lexeme pointed-transliterated (B.:- R;>CIJT B.@R@> >:ELOH ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/g_lex_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/g_lex_utf8.tf">g_lex_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ lexeme pointed-Hebrew (בְּ רֵאשִׁית בָּרָא אֱלֹה)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/g_word" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/g_word.tf">g_word</a>
</div>
<div class="fmono">str</div>

<span> ✅ word pointed-transliterated (B.:- R;>CI73JT B.@R@74> >:ELOHI92JM)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/g_word_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/g_word_utf8.tf">g_word_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ word pointed-Hebrew (בְּ רֵאשִׁ֖ית בָּרָ֣א אֱלֹהִ֑ים)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/gloss" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/gloss.tf">gloss</a>
</div>
<div class="fmono">str</div>

<span> 🆗 english translation of lexeme (beginning create god(s))</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/gn" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/gn.tf">gn</a>
</div>
<div class="fmono">str</div>

<span> ✅ grammatical gender (m; f; NA; unknown.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/label" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/label.tf">label</a>
</div>
<div class="fmono">str</div>

<span> ✅ (half-)verse label (half verses: A; B; C; verses:  GEN 01,02)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/language" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/language.tf">language</a>
</div>
<div class="fmono">str</div>

<span> ✅ of word or lexeme (Hebrew; Aramaic.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/lex" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/lex.tf">lex</a>
</div>
<div class="fmono">str</div>

<span> ✅ lexeme consonantal-transliterated (B R>CJT/ BR>[ >LHJM/)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/lex_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/lex_utf8.tf">lex_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ lexeme consonantal-Hebrew (ב ראשׁית֜ ברא אלהים֜)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/ls" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/ls.tf">ls</a>
</div>
<div class="fmono">str</div>

<span> ✅ lexical set, subclassification of part-of-speech (card; ques; mult)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/nametype" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/nametype.tf">nametype</a>
</div>
<div class="fmono">str</div>

<span> ⚠️ named entity type (pers; mens; gens; topo; ppde.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/nme" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/nme.tf">nme</a>
</div>
<div class="fmono">str</div>

<span> ✅ nominal ending consonantal-transliterated (absent; n/a; JM, ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/nu" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/nu.tf">nu</a>
</div>
<div class="fmono">str</div>

<span> ✅ grammatical number (sg; du; pl; NA; unknown.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/number" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/number.tf">number</a>
</div>
<div class="fmono">int</div>

<span> ✅ sequence number of an object within its context</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/otype" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/otype.tf">otype</a>
</div>
<div class="fmono">str</div>

<span> </span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/pargr" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/pargr.tf">pargr</a>
</div>
<div class="fmono">str</div>

<span> 🆗 hierarchical paragraph number (1; 1.2; 1.2.3.4; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/pdp" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/pdp.tf">pdp</a>
</div>
<div class="fmono">str</div>

<span> ✅ phrase dependent part-of-speech (art; verb; subs; nmpr, ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/pfm" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/pfm.tf">pfm</a>
</div>
<div class="fmono">str</div>

<span> ✅ preformative consonantal-transliterated (absent; n/a; J, ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/prs" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/prs.tf">prs</a>
</div>
<div class="fmono">str</div>

<span> ✅ pronominal suffix consonantal-transliterated (absent; n/a; W; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/prs_gn" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/prs_gn.tf">prs_gn</a>
</div>
<div class="fmono">str</div>

<span> ✅ pronominal suffix gender (m; f; NA; unknown.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/prs_nu" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/prs_nu.tf">prs_nu</a>
</div>
<div class="fmono">str</div>

<span> ✅ pronominal suffix number (sg; du; pl; NA; unknown.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/prs_ps" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/prs_ps.tf">prs_ps</a>
</div>
<div class="fmono">str</div>

<span> ✅ pronominal suffix person (p1; p2; p3; NA; unknown.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/ps" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/ps.tf">ps</a>
</div>
<div class="fmono">str</div>

<span> ✅ grammatical person (p1; p2; p3; NA; unknown.)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/qere" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/qere.tf">qere</a>
</div>
<div class="fmono">str</div>

<span> ✅ word pointed-transliterated masoretic reading correction</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/qere_trailer" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/qere_trailer.tf">qere_trailer</a>
</div>
<div class="fmono">str</div>

<span> ✅ interword material -pointed-transliterated (Masoretic correction)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/qere_trailer_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/qere_trailer_utf8.tf">qere_trailer_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ interword material -pointed-transliterated (Masoretic correction)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/qere_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/qere_utf8.tf">qere_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ word pointed-Hebrew masoretic reading correction</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/rank_lex" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/rank_lex.tf">rank_lex</a>
</div>
<div class="fmono">int</div>

<span> ✅ ranking of lexemes based on freqnuecy</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/rela" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/rela.tf">rela</a>
</div>
<div class="fmono">str</div>

<span> ✅ linguistic relation between clause/(sub)phrase(atom) (ADJ; MOD; ATR; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/sp" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/sp.tf">sp</a>
</div>
<div class="fmono">str</div>

<span> ✅ part-of-speech (art; verb; subs; nmpr, ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/st" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/st.tf">st</a>
</div>
<div class="fmono">str</div>

<span> ✅ state of a noun (a (absolute); c (construct); e (emphatic).)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/tab" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/tab.tf">tab</a>
</div>
<div class="fmono">int</div>

<span> ✅ clause atom: its level in the linguistic embedding</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/trailer" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/trailer.tf">trailer</a>
</div>
<div class="fmono">str</div>

<span> ✅ interword material pointed-transliterated (& 00 05 00_P ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/trailer_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/trailer_utf8.tf">trailer_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ interword material pointed-Hebrew (־ ׃)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/txt" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/txt.tf">txt</a>
</div>
<div class="fmono">str</div>

<span> ✅ text type of clause and surrounding (repetion of ? N D Q as in feature domain)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/typ" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/typ.tf">typ</a>
</div>
<div class="fmono">str</div>

<span> ✅ clause/phrase(atom) type (VP; NP; Ellp; Ptcp; WayX)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/uvf" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/uvf.tf">uvf</a>
</div>
<div class="fmono">str</div>

<span> ✅ univalent final consonant consonantal-transliterated (absent; N; J; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/vbe" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/vbe.tf">vbe</a>
</div>
<div class="fmono">str</div>

<span> ✅ verbal ending consonantal-transliterated (n/a; W; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/vbs" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/vbs.tf">vbs</a>
</div>
<div class="fmono">str</div>

<span> ✅ root formation consonantal-transliterated (absent; n/a; H; ...)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/verse" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/verse.tf">verse</a>
</div>
<div class="fmono">int</div>

<span> ✅ verse number</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/voc_lex" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/voc_lex.tf">voc_lex</a>
</div>
<div class="fmono">str</div>

<span> ✅ vocalized lexeme pointed-transliterated (B.: R;>CIJT BR> >:ELOHIJM)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/voc_lex_utf8" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/voc_lex_utf8.tf">voc_lex_utf8</a>
</div>
<div class="fmono">str</div>

<span> ✅ vocalized lexeme pointed-Hebrew (בְּ רֵאשִׁית ברא אֱלֹהִים)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/vs" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/vs.tf">vs</a>
</div>
<div class="fmono">str</div>

<span> ✅ verbal stem (qal; piel; hif; apel; pael)</span>

</div>

<div class="frow">
    <div class="fnamecat ">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/vt" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/vt.tf">vt</a>
</div>
<div class="fmono">str</div>

<span> ✅ verbal tense (perf; impv; wayq; infc)</span>

</div>

<div class="frow">
    <div class="fnamecat edge">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/mother" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/mother.tf">mother</a>
</div>
<div class="fmono">none</div>

<span> ✅ linguistic dependency between textual objects</span>

</div>

<div class="frow">
    <div class="fnamecat edge">
<a target="_blank" href="https://etcbc.github.io/bhsa/features/oslots" title="~/text-fabric-data/github/etcbc/bhsa/tf/2021/oslots.tf">oslots</a>
</div>
<div class="fmono">none</div>

<span> </span>

</div>

    </div>
</details>

            <b>Settings:</b><br><details ><summary><b>specified</b></summary><ol><li><b>apiVersion</b>: <code>3</code></li><li><b>appName</b>: <code>etcbc/bhsa</code></li><li><b>appPath</b>: <code>/Users/oliverglanz/text-fabric-data/github/etcbc/bhsa/app</code></li><li><b>commit</b>: <code>gb112c161cfd21eae403d51a2733740d8743460e7</code></li><li><b>css</b>: <code>''</code></li><li><details><summary><b>dataDisplay</b>:</summary><ul><li><details><summary><b>exampleSectionHtml</b>:</summary><code>&lt;code&gt;Genesis 1:1&lt;/code&gt; (use &lt;a href="https://github.com/{org}/{repo}/blob/master/tf/{version}/book%40en.tf" target="_blank"&gt;English book names&lt;/a&gt;)</code></details></li><li><details><summary><b>excludedFeatures</b>:</summary><ul><li><code>g_uvf_utf8</code></li><li><code>g_vbs</code></li><li><code>kq_hybrid</code></li><li><code>languageISO</code></li><li><code>g_nme</code></li><li><code>lex0</code></li><li><code>is_root</code></li><li><code>g_vbs_utf8</code></li><li><code>g_uvf</code></li><li><code>dist</code></li><li><code>root</code></li><li><code>suffix_person</code></li><li><code>g_vbe</code></li><li><code>dist_unit</code></li><li><code>suffix_number</code></li><li><code>distributional_parent</code></li><li><code>kq_hybrid_utf8</code></li><li><code>crossrefSET</code></li><li><code>instruction</code></li><li><code>g_prs</code></li><li><code>lexeme_count</code></li><li><code>rank_occ</code></li><li><code>g_pfm_utf8</code></li><li><code>freq_occ</code></li><li><code>crossrefLCS</code></li><li><code>functional_parent</code></li><li><code>g_pfm</code></li><li><code>g_nme_utf8</code></li><li><code>g_vbe_utf8</code></li><li><code>kind</code></li><li><code>g_prs_utf8</code></li><li><code>suffix_gender</code></li><li><code>mother_object_type</code></li></ul></details></li><li><details><summary><b>noneValues</b>:</summary><ul><li><code>absent</code></li><li><code>n/a</code></li><li><code>none</code></li><li><code>unknown</code></li><li><i>no value</i></li><li><code>NA</code></li></ul></details></li></ul></details></li><li><details><summary><b>docs</b>:</summary><ul><li><b>docBase</b>: <code>{docRoot}/{repo}</code></li><li><b>docExt</b>: <code>''</code></li><li><b>docPage</b>: <code>''</code></li><li><b>docRoot</b>: <code>https://{org}.github.io</code></li><li><b>featurePage</b>: <code>0_home</code></li></ul></details></li><li><b>interfaceDefaults</b>: <code>{}</code></li><li><b>isCompatible</b>: <code>True</code></li><li><b>local</b>: <code>local</code></li><li><b>localDir</b>: <code>/Users/oliverglanz/text-fabric-data/github/etcbc/bhsa/_temp</code></li><li><details><summary><b>provenanceSpec</b>:</summary><ul><li><b>corpus</b>: <code>BHSA = Biblia Hebraica Stuttgartensia Amstelodamensis</code></li><li><b>doi</b>: <code>10.5281/zenodo.1007624</code></li><li><b>extraData</b>: <code>ner</code></li><li><details><summary><b>moduleSpecs</b>:</summary><ul><li><details><summary>:</summary><ul><li><b>backend</b>: <i>no value</i></li><li><b>corpus</b>: <code>Phonetic Transcriptions</code></li><li><details><summary><b>docUrl</b>:</summary><code>https://nbviewer.jupyter.org/github/etcbc/phono/blob/master/programs/phono.ipynb</code></details></li><li><b>doi</b>: <code>10.5281/zenodo.1007636</code></li><li><b>org</b>: <code>etcbc</code></li><li><b>relative</b>: <code>/tf</code></li><li><b>repo</b>: <code>phono</code></li></ul></details></li><li><details><summary>:</summary><ul><li><b>backend</b>: <i>no value</i></li><li><b>corpus</b>: <code>Parallel Passages</code></li><li><details><summary><b>docUrl</b>:</summary><code>https://nbviewer.jupyter.org/github/etcbc/parallels/blob/master/programs/parallels.ipynb</code></details></li><li><b>doi</b>: <code>10.5281/zenodo.1007642</code></li><li><b>org</b>: <code>etcbc</code></li><li><b>relative</b>: <code>/tf</code></li><li><b>repo</b>: <code>parallels</code></li></ul></details></li></ul></details></li><li><b>org</b>: <code>etcbc</code></li><li><b>relative</b>: <code>/tf</code></li><li><b>repo</b>: <code>bhsa</code></li><li><b>version</b>: <code>2021</code></li><li><b>webBase</b>: <code>https://shebanq.ancient-data.org/hebrew</code></li><li><b>webHint</b>: <code>Show this on SHEBANQ</code></li><li><b>webLang</b>: <code>la</code></li><li><b>webLexId</b>: <code>True</code></li><li><details><summary><b>webUrl</b>:</summary><code>{webBase}/text?book=&lt;1&gt;&amp;chapter=&lt;2&gt;&amp;verse=&lt;3&gt;&amp;version={version}&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt</code></details></li><li><b>webUrlLex</b>: <code>{webBase}/word?version={version}&amp;id=&lt;lid&gt;</code></li></ul></details></li><li><b>release</b>: <i>no value</i></li><li><details><summary><b>typeDisplay</b>:</summary><ul><li><details><summary><b>clause</b>:</summary><ul><li><b>label</b>: <code>{typ} {rela}</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>clause_atom</b>:</summary><ul><li><b>hidden</b>: <code>True</code></li><li><b>label</b>: <code>{code}</code></li><li><b>level</b>: <code>1</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>half_verse</b>:</summary><ul><li><b>hidden</b>: <code>True</code></li><li><b>label</b>: <code>{label}</code></li><li><b>style</b>: <code>''</code></li><li><b>verselike</b>: <code>True</code></li></ul></details></li><li><details><summary><b>lex</b>:</summary><ul><li><b>featuresBare</b>: <code>gloss</code></li><li><b>label</b>: <code>{voc_lex_utf8}</code></li><li><b>lexOcc</b>: <code>word</code></li><li><b>style</b>: <code>orig</code></li><li><b>template</b>: <code>{voc_lex_utf8}</code></li></ul></details></li><li><details><summary><b>phrase</b>:</summary><ul><li><b>label</b>: <code>{typ} {function}</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>phrase_atom</b>:</summary><ul><li><b>hidden</b>: <code>True</code></li><li><b>label</b>: <code>{typ} {rela}</code></li><li><b>level</b>: <code>1</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>sentence</b>:</summary><ul><li><b>label</b>: <code>{number}</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>sentence_atom</b>:</summary><ul><li><b>hidden</b>: <code>True</code></li><li><b>label</b>: <code>{number}</code></li><li><b>level</b>: <code>1</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>subphrase</b>:</summary><ul><li><b>hidden</b>: <code>True</code></li><li><b>label</b>: <code>{number}</code></li><li><b>style</b>: <code>''</code></li></ul></details></li><li><details><summary><b>word</b>:</summary><ul><li><b>features</b>: <code>pdp vs vt</code></li><li><b>featuresBare</b>: <code>lex:gloss</code></li></ul></details></li></ul></details></li><li><b>writing</b>: <code>hbo</code></li></ol></details>




<style>tr.tf.ltr, td.tf.ltr, th.tf.ltr { text-align: left ! important;}
tr.tf.rtl, td.tf.rtl, th.tf.rtl { text-align: right ! important;}
@font-face {
  font-family: "Gentium Plus";
  src: local('Gentium Plus'), local('GentiumPlus'),
    url('/browser/static/fonts/GentiumPlus-R.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/GentiumPlus-R.woff?raw=true') format('woff');
}

@font-face {
  font-family: "Ezra SIL";
  src: local('Ezra SIL'), local('EzraSIL'),
    url('/browser/static/fonts/SILEOT.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/SILEOT.woff?raw=true') format('woff');
}

@font-face {
  font-family: "SBL Hebrew";
  src: local('SBL Hebrew'), local('SBLHebrew'),
    url('/browser/static/fonts/SBL_Hbrw.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/SBL_Hbrw.woff?raw=true') format('woff');
}

@font-face {
  font-family: "Estrangelo Edessa";
  src: local('Estrangelo Edessa'), local('EstrangeloEdessa');
    url('/browser/static/fonts/SyrCOMEdessa.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/SyrCOMEdessa.woff?raw=true') format('woff');
}

@font-face {
  font-family: AmiriQuran;
  font-style: normal;
  font-weight: 400;
  src: local('Amiri Quran'), local('AmiriQuran'),
    url('/browser/static/fonts/AmiriQuran.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/AmiriQuran.woff?raw=true') format('woff');
}

@font-face {
  font-family: AmiriQuranColored;
  font-style: normal;
  font-weight: 400;
  src: local('Amiri Quran Colored'), local('AmiriQuranColored'),
    url('/browser/static/fonts/AmiriQuranColored.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/AmiriQuranColored.woff?raw=true') format('woff');
}

@font-face {
  font-family: "Santakku";
  src: local('Santakku'),
    url('/browser/static/fonts/Santakku.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/Santakku.woff?raw=true') format('woff');
}

@font-face {
  font-family: "SantakkuM";
  src: local('SantakkuM'),
    url('/browser/static/fonts/SantakkuM.woff') format('woff'),
    url('https://github.com/annotation/text-fabric/blob/master/tf/browser/static/fonts/SantakkuM.woff?raw=true') format('woff');
}

/* bypassing some classical notebook settings */
div#notebook {
  line-height: unset;
}
/* neutral text */
.txtn,.txtn a:visited,.txtn a:link {
    font-family: sans-serif;
    font-size: medium;
    direction: ltr;
    unicode-bidi: embed;
    text-decoration: none;
    color: var(--text-color);
}
/* transcription text */
.txtt,.txtt a:visited,.txtt a:link {
    font-family: monospace;
    font-size: medium;
    direction: ltr;
    unicode-bidi: embed;
    text-decoration: none;
    color: var(--text-color);
}
/* source text */
.txto,.txto a:visited,.txto a:link {
    font-family: serif;
    font-size: medium;
    direction: ltr;
    unicode-bidi: embed;
    text-decoration: none;
    color: var(--text-color);
}
/* phonetic text */
.txtp,.txtp a:visited,.txtp a:link {
    font-family: Gentium, sans-serif;
    font-size: medium;
    direction: ltr;
    unicode-bidi: embed;
    text-decoration: none;
    color: var(--text-color);
}
/* original script text */
.txtu,.txtu a:visited,.txtu a:link {
    font-family: Gentium, sans-serif;
    font-size: medium;
    text-decoration: none;
    color: var(--text-color);
}
/* hebrew */
.txtu.hbo,.lex.hbo {
    font-family: "Ezra SIL", "SBL Hebrew", sans-serif;
    font-size: large;
    direction: rtl ! important;
    unicode-bidi: embed;
}
/* syriac */
.txtu.syc,.lex.syc {
    font-family: "Estrangelo Edessa", sans-serif;
    font-size: medium;
    direction: rtl ! important;
    unicode-bidi: embed;
}
/* neo aramaic */
.txtu.cld,.lex.cld {
    font-family: "CharisSIL-R", sans-serif;
    font-size: medium;
    direction: ltr ! important;
    unicode-bidi: embed;
}
/* standard arabic */
.txtu.ara,.lex.ara {
    font-family: "AmiriQuran", sans-serif;
    font-size: large;
    direction: rtl ! important;
    unicode-bidi: embed;
}
/* cuneiform */
.txtu.akk,.lex.akk {
    font-family: Santakku, sans-serif;
    font-size: large;
    direction: ltr ! important;
    unicode-bidi: embed;
}
.txtu.uga,.lex.uga {
    font-family: sans-serif;
    /*font-family: Noto Sans Ugaritic, sans-serif;*/
    font-size: large;
    direction: ltr ! important;
    unicode-bidi: embed;
}
/* greek */
.txtu.grc,.lex.grc a:link {
    font-family: Gentium, sans-serif;
    font-size: medium;
    direction: ltr ! important;
    unicode-bidi: embed;
}
a:hover {
    text-decoration: underline | important;
    color: #0000ff | important;
}
.ltr {
    direction: ltr ! important;
    unicode-bidi: embed;
}
.rtl {
    direction: rtl ! important;
    unicode-bidi: embed;
}
.ubd {
    unicode-bidi: embed;
}
.col {
   display: inline-block;
}
.features {
    font-family: monospace;
    font-size: medium;
    font-weight: bold;
    color: var(--features);
    display: flex;
    flex-flow: column nowrap;
    justify-content: flex-start;
    align-items: flex-start;
    align-content: flex-start;
    padding: 2px;
    margin: 2px;
    direction: ltr;
    unicode-bidi: embed;
    border: var(--meta-width) solid var(--meta-color);
    border-radius: var(--meta-width);
}
.features div,.features span {
    padding: 0;
    margin: -2px 0;
}
.features .f {
    font-family: sans-serif;
    font-size: small;
    font-weight: normal;
    color: #5555bb;
}
.features .xft {
  color: #000000;
  background-color: #eeeeee;
  font-size: medium;
  margin: 2px 0px;
}
.features .xft .f {
  color: #000000;
  background-color: #eeeeee;
  font-size: small;
  font-weight: normal;
}
.tfsechead {
    font-family: sans-serif;
    font-size: small;
    font-weight: bold;
    color: var(--tfsechead);
    unicode-bidi: embed;
    text-align: start;
}
.structure {
    font-family: sans-serif;
    font-size: small;
    font-weight: bold;
    color: var(--structure);
    unicode-bidi: embed;
    text-align: start;
}
.comments {
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    align-content: flex-start;
    flex-flow: column nowrap;
}
.nd, a:link.nd {
    font-family: sans-serif;
    font-size: small;
    color: var(--node);
    vertical-align: super;
    direction: ltr ! important;
    unicode-bidi: embed;
}
.nde, a:link.nde {
    font-family: sans-serif;
    font-size: small;
    color: var(--node);
    direction: ltr ! important;
    unicode-bidi: embed;
}
.etf {
    font-size: normal;
    border-radius: 0.2rem;
    border: 1pt solid white;
    padding: 0 0.2rem ! important;
    margin: 0 0.2rem ! important;
}
.etfx {
    font-size: x-large;
}
.lex {
  color: var(--lex-color);;
}
#colormapplus, #colormapmin, .ecolormapmin {
  font-weight: bold;
  border-radius: 0.1rem;
  background-color: #eeeeff;
  padding: 0 1rem;
  margin: 0 1rem;
}
.clr {
  font-style: italic;
  font-size: small;
}
.clmap,.eclmap {
  padding: 0;
}
.children,.children.ltr {
    display: flex;
    border: 0;
    background-color: #ffffff;
    justify-content: flex-start;
    align-items: flex-start;
    align-content: flex-start;
}
.children.stretch {
    align-items: stretch;
}
.children.hor {
    flex-flow: row nowrap;
}
.children.hor.wrap {
    flex-flow: row wrap;
}
.children.ver {
    flex-flow: column nowrap;
}
.children.ver.wrap {
    flex-flow: column wrap;
}
.contnr {
    width: fit-content;
    display: flex;
    justify-content: flex-start;
    align-items: flex-start;
    align-content: flex-start;
    flex-flow: column nowrap;
    background: #ffffff none repeat scroll 0 0;
    padding:  10px 2px 2px 2px;
    margin: 16px 2px 2px 2px;
    border-style: solid;
    font-size: small;
}
.contnr.trm {
    background-attachment: local;
}
.contnr.cnul {
    padding:  0;
    margin: 0;
    border-style: solid;
    font-size: xx-small;
}
.contnr.cnul,.lbl.cnul {
    border-color: var(--border-color-nul);
    border-width: var(--border-width-nul);
    border-radius: var(--border-width-nul);
}
.contnr.c0,.lbl.c0 {
    border-color: var(--border-color0);
    border-width: var(--border-width0);
    border-radius: var(--border-width0);
}
.contnr.c1,.lbl.c1 {
    border-color: var(--border-color1);
    border-width: var(--border-width1);
    border-radius: var(--border-width1);
}
.contnr.c2,.lbl.c2 {
    border-color: var(--border-color2);
    border-width: var(--border-width2);
    border-radius: var(--border-width2);
}
.contnr.c3,.lbl.c3 {
    border-color: var(--border-color3);
    border-width: var(--border-width3);
    border-radius: var(--border-width3);
}
.contnr.c4,.lbl.c4 {
    border-color: var(--border-color4);
    border-width: var(--border-width4);
    border-radius: var(--border-width4);
}
span.plain {
    /*display: inline-block;*/
    display: inline-flex;
    flex-flow: row wrap;
    white-space: pre-wrap;
}
span.break {
  flex-basis: 100%;
  height: 0;
}
.plain {
    background-color: #ffffff;
}
.plain.l,.contnr.l,.contnr.l>.lbl {
    border-left-style: dotted
}
.plain.r,.contnr.r,.contnr.r>.lbl {
    border-right-style: dotted
}
.plain.lno,.contnr.lno,.contnr.lno>.lbl {
    border-left-style: none
}
.plain.rno,.contnr.rno,.contnr.rno>.lbl {
    border-right-style: none
}
.plain.l {
    padding-left: 4px;
    margin-left: 2px;
    border-width: var(--border-width-plain);
}
.plain.r {
    padding-right: 4px;
    margin-right: 2px;
    border-width: var(--border-width-plain);
}
.lbl {
    font-family: monospace;
    margin-top: -24px;
    margin-left: 20px;
    background: #ffffff none repeat scroll 0 0;
    padding: 0 6px;
    border-style: solid;
    display: block;
    color: var(--label)
}
.lbl.trm {
    background-attachment: local;
    margin-top: 2px;
    margin-left: 2px;
    padding: 2px 2px;
    border-style: none;
}
.lbl.cnul {
    font-size: xx-small;
}
.lbl.c0 {
    font-size: small;
}
.lbl.c1 {
    font-size: small;
}
.lbl.c2 {
    font-size: medium;
}
.lbl.c3 {
    font-size: medium;
}
.lbl.c4 {
    font-size: large;
}
.occs, a:link.occs {
    font-size: small;
}

/* PROVENANCE */

div.prov {
	margin: 40px;
	padding: 20px;
	border: 2px solid var(--fog-rim);
}
div.pline {
	display: flex;
	flex-flow: row nowrap;
	justify-content: stretch;
	align-items: baseline;
}
div.p2line {
	margin-left: 2em;
	display: flex;
	flex-flow: row nowrap;
	justify-content: stretch;
	align-items: baseline;
}
div.psline {
	display: flex;
	flex-flow: row nowrap;
	justify-content: stretch;
	align-items: baseline;
	background-color: var(--gold-mist-back);
}
div.pname {
	flex: 0 0 5rem;
	font-weight: bold;
}
div.pval {
    flex: 1 1 auto;
}

/* KEYBOARD */
.ccoff {
  background-color: inherit;
}
.ccon {
  background-color: yellow ! important;
}
.ccon,.ccoff {
  padding: 0.2rem;
  margin: 0.2rem;
  border: 0.1rem solid var(--letter-box-border);
  border-radius: 0.1rem;
}
.ccline {
  font-size: xx-large ! important;
  font-weight: bold;
  line-height: 2em ! important;
}
/* TF header */

summary {
  /* needed to override the normalize.less
   * in the classical Jupyter Notebook
   */
  display: list-item ! important;
}

.fcorpus {
  display: flex;
  flex-flow: column nowrap;
  justify-content: flex-start;
  align-items: flex-start;
  align-content: flex-start;
  overflow: auto;
}
.frow {
  display: flex;
  flex-flow: row nowrap;
  justify-content: flex-start;
  align-items: flex-start;
  align-content: flex-start;
}
.fmeta {
  display: flex;
  flex-flow: column nowrap;
  justify-content: flex-start;
  align-items: flex-start;
  align-content: flex-start;
}
.fmetarow {
  display: flex;
  flex-flow: row nowrap;
  justify-content: flex-start;
  align-items: flex-start;
  align-content: flex-start;
}
.fmetakey {
  min-width: 8em;
  font-family: monospace;
}
.fnamecat {
  min-width: 8em;
}
.fnamecat.edge {
  font-weight: bold;
  font-style: italic;
}
.fmono {
    font-family: monospace;
}

:root {
	--node:               hsla(120, 100%,  20%, 1.0  );
	--label:              hsla(  0, 100%,  20%, 1.0  );
	--tfsechead:          hsla(  0, 100%,  25%, 1.0  );
	--structure:          hsla(120, 100%,  25%, 1.0  );
	--features:           hsla(  0,   0%,  30%, 1.0  );
  --text-color:         hsla( 60,  80%,  10%, 1.0  );
  --lex-color:          hsla(220,  90%,  60%, 1.0  );
  --meta-color:         hsla(  0,   0%,  90%, 0.7  );
  --meta-width:         3px;
  --border-color-nul:   hsla(  0,   0%,  90%, 0.5  );
  --border-color0:      hsla(  0,   0%,  90%, 0.9  );
  --border-color1:      hsla(  0,   0%,  80%, 0.9  );
  --border-color2:      hsla(  0,   0%,  70%, 0.9  );
  --border-color3:      hsla(  0,   0%,  80%, 0.8  );
  --border-color4:      hsla(  0,   0%,  60%, 0.9  );
	--letter-box-border:  hsla(  0,   0%,  80%, 0.5  );
  --border-width-nul:   2px;
  --border-width0:      2px;
  --border-width1:      3px;
  --border-width2:      4px;
  --border-width3:      6px;
  --border-width4:      5px;
  --border-width-plain: 2px;
}
.hl {
  background-color: var(--hl-strong);
}
span.hl {
	background-color: var(--hl-strong);
	border-width: 0;
	border-radius: 2px;
	border-style: solid;
}
div.contnr.hl,div.lbl.hl {
  background-color: var(--hl-strong);
}
div.contnr.hl {
  border-color: var(--hl-rim) ! important;
	border-width: 4px ! important;
}

span.hlbx {
	border-color: var(--hl-rim);
	border-width: 4px ! important;
	border-style: solid;
	border-radius: 6px;
  padding: 4px;
  margin: 4px;
}
.ehl {
  background-color: var(--ehl-strong);
}

:root {
	--hl-strong:        hsla( 60, 100%,  70%, 0.9  );
	--hl-rim:           hsla( 55,  80%,  50%, 1.0  );
	--ehl-strong:       hsla(240, 100%,  70%, 0.9  );
}
</style>




<script>
globalThis.copyChar = (el, c) => {
    for (const el of document.getElementsByClassName('ccon')) {
        el.className = 'ccoff'
    }
    el.className = 'ccon'
    navigator.clipboard.writeText(String.fromCharCode(c))
}
</script>




<div><b>TF API:</b> names <a target="_blank" href="https://annotation.github.io/text-fabric/tf/cheatsheet.html" title="doc">N F E L T S C TF Fs Fall Es Eall Cs Call</a> directly usable</div><hr>


    CPU times: user 4.41 s, sys: 562 ms, total: 4.97 s
    Wall time: 5.17 s


# Data Generation

## Any Valence of JR>


```python
FearValence1_any = '''
verse verse book chapter
  clause
    phrase function typ
      word lex=JR>[ vs vt lex_utf8 gn nu ps sp macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
    phrase function typ
      word lex lex_utf8 vs vt gn nu ps sp macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
'''
FearValence1_any  = BHS.search(FearValence1_any)
BHS.table(FearValence1_any, start=1, end=3, condensed=False, extraFeatures='qere_utf8', fmt='text-orig-full-ketiv', colorMap={3:'yellow', 4:'magenta'})
```

      1.43s 1413 results



<table><tr class="tf rtl"><th class="tf rtl">n</th><th class="tf rtl">p</th><th class="tf rtl">verse</th><th class="tf">clause</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">phrase</th><th class="tf">word</th></tr>
<tr class="tf rtl"><td class="tf rtl">1</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Genesis&amp;chapter=3&amp;verse=10&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Genesis 3:10">Genesis 3:10</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">וָ</span></span></span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">וָ</span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">וָ</span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">2</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Genesis&amp;chapter=3&amp;verse=10&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Genesis 3:10">Genesis 3:10</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וָ</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">אִירָ֛א </span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">3</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Genesis&amp;chapter=15&amp;verse=1&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Genesis 15:1">Genesis 15:1</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אַל־</span></span></span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֣א </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֣א </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֣א </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אַל־</span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אַל־</span></span></span></td></tr></table>



```python
BHS.export(FearValence1_any, toDir='/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export', toFile='FearValDataExport_01_all.tsv')
```


```python
FearValData_all=pd.read_csv('/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValDataExport_01_all.tsv',delimiter='\t',encoding='utf-16')
pd.set_option('display.max_columns', 100)
FearValData_all.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>NODE1</th>
      <th>TYPE1</th>
      <th>TEXT1</th>
      <th>book1</th>
      <th>chapter1</th>
      <th>verse1</th>
      <th>NODE2</th>
      <th>TYPE2</th>
      <th>TEXT2</th>
      <th>NODE3</th>
      <th>TYPE3</th>
      <th>TEXT3</th>
      <th>function3</th>
      <th>typ3</th>
      <th>NODE4</th>
      <th>TYPE4</th>
      <th>TEXT4</th>
      <th>gn4</th>
      <th>lex4</th>
      <th>lex_utf84</th>
      <th>macula_BH_lxx_lex_14</th>
      <th>macula_BH_lxx_lex_24</th>
      <th>macula_BH_lxx_lex_34</th>
      <th>macula_BH_lxx_text4</th>
      <th>nu4</th>
      <th>ps4</th>
      <th>sp4</th>
      <th>tov_BH_text4</th>
      <th>tov_LXX_lex4</th>
      <th>tov_LXX_text4</th>
      <th>vs4</th>
      <th>vt4</th>
      <th>NODE5</th>
      <th>TYPE5</th>
      <th>TEXT5</th>
      <th>function5</th>
      <th>typ5</th>
      <th>NODE6</th>
      <th>TYPE6</th>
      <th>TEXT6</th>
      <th>gn6</th>
      <th>lex6</th>
      <th>lex_utf86</th>
      <th>macula_BH_lxx_lex_16</th>
      <th>macula_BH_lxx_lex_26</th>
      <th>macula_BH_lxx_lex_36</th>
      <th>macula_BH_lxx_text6</th>
      <th>nu6</th>
      <th>ps6</th>
      <th>sp6</th>
      <th>tov_BH_text6</th>
      <th>tov_LXX_lex6</th>
      <th>tov_LXX_text6</th>
      <th>vs6</th>
      <th>vt6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>1414454</td>
      <td>verse</td>
      <td>וַיֹּ֕אמֶר אֶת־קֹלְךָ֥ שָׁמַ֖עְתִּי בַּגָּ֑ן ו...</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>clause</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>phrase</td>
      <td>אִירָ֛א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>1364</td>
      <td>word</td>
      <td>אִירָ֛א</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβήθην</td>
      <td>sg</td>
      <td>p1</td>
      <td>verb</td>
      <td>אירא</td>
      <td>φοβέω</td>
      <td>ἐφοβήθην</td>
      <td>qal</td>
      <td>wayq</td>
      <td>652348</td>
      <td>phrase</td>
      <td>וָ</td>
      <td>Conj</td>
      <td>CP</td>
      <td>1363</td>
      <td>word</td>
      <td>וָ</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καί</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>καὶ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>conj</td>
      <td>ו</td>
      <td>καί</td>
      <td>καὶ</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>1414454</td>
      <td>verse</td>
      <td>וַיֹּ֕אמֶר אֶת־קֹלְךָ֥ שָׁמַ֖עְתִּי בַּגָּ֑ן ו...</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>clause</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>phrase</td>
      <td>אִירָ֛א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>1364</td>
      <td>word</td>
      <td>אִירָ֛א</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβήθην</td>
      <td>sg</td>
      <td>p1</td>
      <td>verb</td>
      <td>אירא</td>
      <td>φοβέω</td>
      <td>ἐφοβήθην</td>
      <td>qal</td>
      <td>wayq</td>
      <td>652349</td>
      <td>phrase</td>
      <td>אִירָ֛א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>1364</td>
      <td>word</td>
      <td>אִירָ֛א</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβήθην</td>
      <td>sg</td>
      <td>p1</td>
      <td>verb</td>
      <td>אירא</td>
      <td>φοβέω</td>
      <td>ἐφοβήθην</td>
      <td>qal</td>
      <td>wayq</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>1414750</td>
      <td>verse</td>
      <td>אַחַ֣ר׀ הַדְּבָרִ֣ים הָאֵ֗לֶּה הָיָ֤ה דְבַר־יְ...</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>clause</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>phrase</td>
      <td>תִּירָ֣א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>6686</td>
      <td>word</td>
      <td>תִּירָ֣א</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβοῦ</td>
      <td>sg</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא</td>
      <td>φοβέω</td>
      <td>φοβοῦ</td>
      <td>qal</td>
      <td>impf</td>
      <td>655222</td>
      <td>phrase</td>
      <td>אַל־</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>6685</td>
      <td>word</td>
      <td>אַל־</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μή</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>μὴ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>nega</td>
      <td>אל</td>
      <td>μή</td>
      <td>μὴ</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>1414750</td>
      <td>verse</td>
      <td>אַחַ֣ר׀ הַדְּבָרִ֣ים הָאֵ֗לֶּה הָיָ֤ה דְבַר־יְ...</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>clause</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>phrase</td>
      <td>תִּירָ֣א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>6686</td>
      <td>word</td>
      <td>תִּירָ֣א</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβοῦ</td>
      <td>sg</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא</td>
      <td>φοβέω</td>
      <td>φοβοῦ</td>
      <td>qal</td>
      <td>impf</td>
      <td>655223</td>
      <td>phrase</td>
      <td>תִּירָ֣א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>6686</td>
      <td>word</td>
      <td>תִּירָ֣א</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβοῦ</td>
      <td>sg</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא</td>
      <td>φοβέω</td>
      <td>φοβοῦ</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>1414828</td>
      <td>verse</td>
      <td>וַתְּכַחֵ֨שׁ שָׂרָ֧ה׀ לֵאמֹ֛ר לֹ֥א צָחַ֖קְתִּי...</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>clause</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>phrase</td>
      <td>יָרֵ֑אָה</td>
      <td>Pred</td>
      <td>VP</td>
      <td>8069</td>
      <td>word</td>
      <td>יָרֵ֑אָה</td>
      <td>f</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβήθη</td>
      <td>sg</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראה</td>
      <td>φοβέω</td>
      <td>ἐφοβήθη</td>
      <td>qal</td>
      <td>perf</td>
      <td>656101</td>
      <td>phrase</td>
      <td>כִּ֣י׀</td>
      <td>Conj</td>
      <td>CP</td>
      <td>8068</td>
      <td>word</td>
      <td>כִּ֣י׀</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>conj</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename columns
FearValData_all = FearValData_all.rename(columns={
"R":"order",
"book1":"book",
"chapter1":"chapter",
"verse1":"verse",
"NODE2":"clause #",
"TEXT2":"verse_text",



"NODE3":"phrase # Pred",
"function3":"function Pred",
"typ3":"phrase type Pred",
"TEXT3":"phrase text Pred",

"NODE4":"word H # Pred",
"TEXT4":"word H text Pred",
"sp4":"word H Pred part of speech",
"nu4":"word H Pred number",
"ps4":"word H Pred person",
"gn4":"word H Pred gender",
"vs4":"word H Pred stem",
"vt4":"word H Pred tense",
"lex4":"word H lex Pred",
"lex_utf84":"word H lexUTF8 Pred",
"tov_LXX_text4":"word G text Pred",
"tov_LXX_lex4":"word G lex Pred",



"NODE5":"phrase # Cmpl",
"function5":"function Cmpl",
"typ5":"phrase type Cmpl",
"TEXT5":"phrase text Cmpl",

"NODE6":"word H # Cmpl-element-1",
"TEXT6":"word H text Cmpl-element-1",
"sp6":"word H Cmpl-element-1 part of speech",
"nu6":"word H Cmpl-element-1 number",
"ps6":"word H Cmpl-element-1 person",
"gn6":"word H Cmpl-element-1 gender",
"vs6":"word H Cmpl-element-1 stem",
"vt6":"word H Cmpl-element-1 tense",
"lex6":"word H lex Cmpl-element-1",
"lex_utf86":"word H lexUTF8 Cmpl-element-1",
"tov_LXX_text6":"word G text Cmpl-element-1",
"tov_LXX_lex6":"word G lex Cmpl-element-1",

"NODE7":"word H # Cmpl-element-2",
"TEXT7":"word H text Cmpl-element-2",
"sp7":"word H Cmpl-element-2 part of speech",
"nu7":"word H Cmpl-element-2 number",
"ps7":"word H Cmpl-element-2 person",
"gn7":"word H Cmpl-element-2 gender",
"vs7":"word H Cmpl-element-2 stem",
"vt7":"word H Cmpl-element-2 tense",
"lex7":"word H lex Cmpl-element-2",
"lex_utf87":"word H lexUTF8 Cmpl-element-2",
"tov_LXX_text7":"word G text Cmpl-element-2",
"tov_LXX_lex7":"word G lex Cmpl-element-2"
})

# Reduce DataFrame to the selected columns in the desired order
FearValData_all = FearValData_all[
    [
"order",
"book",
"chapter",
"verse",
"clause #",
"verse_text",


"phrase # Pred",
"function Pred",
"phrase type Pred",
"phrase text Pred",

"word H # Pred",
"word H text Pred",
"word H Pred part of speech",
"word H Pred number",
"word H Pred person",
"word H Pred gender",
"word H Pred stem",
"word H Pred tense",
"word H lex Pred",
"word H lexUTF8 Pred",
"word G text Pred",
"word G lex Pred",




"phrase # Cmpl",
"function Cmpl",
"phrase type Cmpl",
"phrase text Cmpl",

"word H # Cmpl-element-1",
"word H text Cmpl-element-1",
"word H Cmpl-element-1 part of speech",
"word H Cmpl-element-1 number",
"word H Cmpl-element-1 person",
"word H Cmpl-element-1 gender",
"word H Cmpl-element-1 stem",
"word H Cmpl-element-1 tense",
"word H lex Cmpl-element-1",
"word H lexUTF8 Cmpl-element-1",
"word G text Cmpl-element-1",
"word G lex Cmpl-element-1",

    ]
]
FearValData_all.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652348</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וָ</td>
      <td>1363</td>
      <td>וָ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καὶ</td>
      <td>καί</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655222</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>אַל־</td>
      <td>6685</td>
      <td>אַל־</td>
      <td>nega</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μὴ</td>
      <td>μή</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656101</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֣י׀</td>
      <td>8068</td>
      <td>כִּ֣י׀</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValData_all.to_excel("/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValResearchData_00_any.xlsx", index=False)

```


```python
import re
import unicodedata

# ----------------------------
# HEBREW
# ----------------------------

# Shin/sin dots
SHIN_DOT = "\u05C1"  # שׁ
SIN_DOT  = "\u05C2"  # שׂ

# Basic consonant mapping (SBL-style)
HEB_TO_SBL = {
    "א": "ʾ",
    "ב": "b",
    "ג": "g",
    "ד": "d",
    "ה": "h",
    "ו": "w",
    "ז": "z",
    "ח": "ḥ",
    "ט": "ṭ",
    "י": "y",
    "כ": "k", "ך": "k",
    "ל": "l",
    "מ": "m", "ם": "m",
    "נ": "n", "ן": "n",
    "ס": "s",
    "ע": "ʿ",
    "פ": "p", "ף": "p",
    "צ": "ṣ", "ץ": "ṣ",
    "ק": "q",
    "ר": "r",
    "ת": "t",
    # "ש" handled specially below
}

# remove Hebrew vowel points/cantillation
HEBREW_MARKS_RE = re.compile(r"[\u0591-\u05BD\u05BF-\u05C7]")

def translit_lexeme_sbl(s: str) -> str:
    if s is None:
        return s
    s = str(s)

    # Handle שׁ / שׂ explicitly
    s = s.replace("ש" + SHIN_DOT, "Š")  # temp token
    s = s.replace("ש" + SIN_DOT,  "Ś")  # temp token

    # Drop remaining Hebrew marks
    s = HEBREW_MARKS_RE.sub("", s)

    out = []
    for ch in s:
        if ch == "ש":
            out.append("š")     # default when dot is absent
        elif ch == "Š":
            out.append("š")
        elif ch == "Ś":
            out.append("ś")
        else:
            out.append(HEB_TO_SBL.get(ch, ch))
    return "".join(out)


# ----------------------------
# GREEK
# ----------------------------

GREEK_TO_SBL = {
    "α": "a", "Α": "A",
    "β": "b", "Β": "B",
    "γ": "g", "Γ": "G",
    "δ": "d", "Δ": "D",
    "ε": "e", "Ε": "E",
    "ζ": "z", "Ζ": "Z",
    "η": "ē", "Η": "Ē",
    "θ": "th", "Θ": "Th",
    "ι": "i", "Ι": "I",
    "κ": "k", "Κ": "K",
    "λ": "l", "Λ": "L",
    "μ": "m", "Μ": "M",
    "ν": "n", "Ν": "N",
    "ξ": "x", "Ξ": "X",
    "ο": "o", "Ο": "O",
    "π": "p", "Π": "P",
    "ρ": "r", "Ρ": "R",
    "σ": "s", "Σ": "S",
    "ς": "s",
    "τ": "t", "Τ": "T",
    "υ": "y", "Υ": "Y",
    "φ": "ph", "Φ": "Ph",
    "χ": "ch", "Χ": "Ch",
    "ψ": "ps", "Ψ": "Ps",
    "ω": "ō", "Ω": "Ō",
}

# Unicode combining marks used in Greek
ROUGH_BREATHING = "\u0314"
SMOOTH_BREATHING = "\u0313"
IOTA_SUBSCRIPT = "\u0345"

def translit_greek_sbl(s: str) -> str:
    if s is None:
        return s
    s = str(s)

    # Decompose combined Greek chars into base + combining marks
    s = unicodedata.normalize("NFD", s)

    out = []
    i = 0
    while i < len(s):
        ch = s[i]

        # Non-base combining marks should normally be skipped here
        if unicodedata.combining(ch):
            i += 1
            continue

        # Collect combining marks following this base letter
        j = i + 1
        marks = []
        while j < len(s) and unicodedata.combining(s[j]):
            marks.append(s[j])
            j += 1

        base = ch

        # Base transliteration
        translit = GREEK_TO_SBL.get(base, base)

        # Rough breathing = prefixed h
        # Example: ἁ -> ha, ἑ -> he, ῥ -> rh
        if ROUGH_BREATHING in marks:
            translit = "h" + translit

        # Ignore:
        # - smooth breathing
        # - accents
        # - diaeresis
        # - circumflex
        # - macron/breve
        # - iota subscript
        # If you want iota subscript rendered, that can be added later.

        out.append(translit)
        i = j

    return "".join(out)


# Create a new transliteration column
FearValData_all["lexeme_hebrew_sbl"] = FearValData_all["word H lexUTF8 Pred"].apply(translit_lexeme_sbl)
FearValData_all.head()

FearValData_all["lexeme_greek_sbl"] = FearValData_all["word G lex Pred"].apply(translit_greek_sbl)
FearValData_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>lexeme_hebrew_sbl</th>
      <th>lexeme_greek_sbl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652348</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וָ</td>
      <td>1363</td>
      <td>וָ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καὶ</td>
      <td>καί</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655222</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>אַל־</td>
      <td>6685</td>
      <td>אַל־</td>
      <td>nega</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μὴ</td>
      <td>μή</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656101</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֣י׀</td>
      <td>8068</td>
      <td>כִּ֣י׀</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValData_all.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>lexeme_hebrew_sbl</th>
      <th>lexeme_greek_sbl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652348</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וָ</td>
      <td>1363</td>
      <td>וָ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καὶ</td>
      <td>καί</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655222</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>אַל־</td>
      <td>6685</td>
      <td>אַל־</td>
      <td>nega</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μὴ</td>
      <td>μή</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656101</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֣י׀</td>
      <td>8068</td>
      <td>כִּ֣י׀</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
  </tbody>
</table>
</div>



### Hebrew Stats


```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"          # <-- make all text bold
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"

# Now plot as usual
stem_counts = FearValData_all["word H Cmpl-element-1 stem"].value_counts().sort_values(ascending=False)

plt.figure()
bars = plt.bar(stem_counts.index, stem_counts.values)

#plt.xlabel("Stem")
plt.ylabel("Count")
plt.title("Distribution of yrʾ by Stem")

for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

# Save BEFORE show
plt.savefig("fig01_jara_stem_distribution.pdf", bbox_inches="tight")
plt.savefig("fig01_jara_stem_distribution.png", dpi=600, bbox_inches="tight")
plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_14_0.png)
    



```python
import pandas as pd
import matplotlib.pyplot as plt

# Define tense groups
finite = {"perf", "impf", "impv", "wayq"}
nonfinite = {"infc", "infa", "ptca", "ptcp"}

# Filter to Qal only
df_qal = FearValData_all[FearValData_all["word H Cmpl-element-1 stem"] == "qal"].copy()

# Count each group
finite_count = df_qal["word H Pred tense"].isin(finite).sum()
nonfinite_count = df_qal["word H Pred tense"].isin(nonfinite).sum()

counts = pd.Series(
    {"finite forms": finite_count,
     "non-finite forms": nonfinite_count}
)

counts

```




    finite forms        250
    non-finite forms     52
    dtype: int64




```python
import matplotlib as mpl
mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"

plt.figure()
bars = plt.bar(counts.index, counts.values)
plt.ylabel("Count")
plt.title("Qal yrʾ: finite vs. non-finite forms")

# add numbers on bars
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        str(int(height)),
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=0, ha="center")
plt.savefig("fig02_jara_qal_finite_vs_nonfinite.pdf", bbox_inches="tight")
plt.savefig("fig02_jara_qal_finite_vs_nonfinite.png", dpi=600, bbox_inches="tight")
plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_16_0.png)
    


#### Finite verbal forms in Qal


```python
FearValData_all.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>lexeme_hebrew_sbl</th>
      <th>lexeme_greek_sbl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652348</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וָ</td>
      <td>1363</td>
      <td>וָ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καὶ</td>
      <td>καί</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655222</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>אַל־</td>
      <td>6685</td>
      <td>אַל־</td>
      <td>nega</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μὴ</td>
      <td>μή</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656101</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֣י׀</td>
      <td>8068</td>
      <td>כִּ֣י׀</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
  </tbody>
</table>
</div>




```python
finite_tenses = {"perf", "impf", "wayq", "impv"}

valid_clauses = set(
    FearValData_all.loc[
        (FearValData_all["word H lex Pred"] == "JR>[") &
        (FearValData_all["word H Pred stem"] == "qal") &
        (FearValData_all["word H Pred tense"].isin(finite_tenses)),
        "clause #"
    ]
)
FearValData_QalFinite = FearValData_all[
    FearValData_all["clause #"].isin(valid_clauses)
].copy()

FearValData_QalFinite.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>lexeme_hebrew_sbl</th>
      <th>lexeme_greek_sbl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652348</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וָ</td>
      <td>1363</td>
      <td>וָ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καὶ</td>
      <td>καί</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655222</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>אַל־</td>
      <td>6685</td>
      <td>אַל־</td>
      <td>nega</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μὴ</td>
      <td>μή</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656101</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֣י׀</td>
      <td>8068</td>
      <td>כִּ֣י׀</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValData_QalFinite.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>lexeme_hebrew_sbl</th>
      <th>lexeme_greek_sbl</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652348</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וָ</td>
      <td>1363</td>
      <td>וָ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>καὶ</td>
      <td>καί</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Genesis</td>
      <td>3</td>
      <td>10</td>
      <td>427818</td>
      <td>וָאִירָ֛א</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>652349</td>
      <td>Pred</td>
      <td>VP</td>
      <td>אִירָ֛א</td>
      <td>1364</td>
      <td>אִירָ֛א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p1</td>
      <td>unknown</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθην</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655222</td>
      <td>Nega</td>
      <td>NegP</td>
      <td>אַל־</td>
      <td>6685</td>
      <td>אַל־</td>
      <td>nega</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;L=</td>
      <td>אל</td>
      <td>μὴ</td>
      <td>μή</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Genesis</td>
      <td>15</td>
      <td>1</td>
      <td>428731</td>
      <td>אַל־תִּירָ֣א</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>655223</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֣א</td>
      <td>6686</td>
      <td>תִּירָ֣א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβοῦ</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656101</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֣י׀</td>
      <td>8068</td>
      <td>כִּ֣י׀</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Genesis</td>
      <td>18</td>
      <td>15</td>
      <td>429019</td>
      <td>כִּ֣י׀ יָרֵ֑אָה</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656102</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֑אָה</td>
      <td>8069</td>
      <td>יָרֵ֑אָה</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Genesis</td>
      <td>19</td>
      <td>30</td>
      <td>429249</td>
      <td>כִּ֥י יָרֵ֖א</td>
      <td>656771</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֖א</td>
      <td>9075</td>
      <td>יָרֵ֖א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656770</td>
      <td>Conj</td>
      <td>CP</td>
      <td>כִּ֥י</td>
      <td>9074</td>
      <td>כִּ֥י</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>KJ</td>
      <td>כי</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Genesis</td>
      <td>19</td>
      <td>30</td>
      <td>429249</td>
      <td>כִּ֥י יָרֵ֖א</td>
      <td>656771</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֖א</td>
      <td>9075</td>
      <td>יָרֵ֖א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>656771</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֖א</td>
      <td>9075</td>
      <td>יָרֵ֖א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Genesis</td>
      <td>20</td>
      <td>8</td>
      <td>429326</td>
      <td>וַיִּֽירְא֥וּ הָאֲנָשִׁ֖ים מְאֹֽד׃</td>
      <td>657007</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יִּֽירְא֥וּ</td>
      <td>9421</td>
      <td>יִּֽירְא֥וּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθησαν</td>
      <td>φοβέω</td>
      <td>657006</td>
      <td>Conj</td>
      <td>CP</td>
      <td>וַ</td>
      <td>9420</td>
      <td>וַ</td>
      <td>conj</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>W</td>
      <td>ו</td>
      <td>δὲ</td>
      <td>δέ</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Genesis</td>
      <td>20</td>
      <td>8</td>
      <td>429326</td>
      <td>וַיִּֽירְא֥וּ הָאֲנָשִׁ֖ים מְאֹֽד׃</td>
      <td>657007</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יִּֽירְא֥וּ</td>
      <td>9421</td>
      <td>יִּֽירְא֥וּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθησαν</td>
      <td>φοβέω</td>
      <td>657007</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יִּֽירְא֥וּ</td>
      <td>9421</td>
      <td>יִּֽירְא֥וּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθησαν</td>
      <td>φοβέω</td>
      <td>yrʾ</td>
      <td>phobeō</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValData_all.shape
```




    (1413, 40)




```python
FearValData_QalFinite.shape
```




    (911, 40)




```python
phrase_functions_of_interest = {"PreO", "Objc", "Cmpl"}

phrase_distribution = (
    FearValData_QalFinite
    .loc[FearValData_QalFinite["function Cmpl"].isin(phrase_functions_of_interest)]
    ["function Cmpl"]
    .value_counts()
)

phrase_distribution
```




    function Cmpl
    Objc    140
    Cmpl    129
    PreO     12
    Name: count, dtype: int64




```python
FearValData_QalFinite["function Cmpl"].value_counts()

```




    function Cmpl
    Pred    236
    Objc    140
    Cmpl    129
    Conj    122
    Nega    121
    Subj     72
    Time     27
    Modi     24
    Adju     14
    PreO     12
    Ques      7
    Rela      3
    Intj      2
    Loca      2
    Name: count, dtype: int64




```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# Styling (as before)
mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Get counts and exclude 'Pred'
func_counts = (
    FearValData_QalFinite["function Cmpl"]
    .value_counts()
    .drop("Pred", errors="ignore")
)

# Plot
plt.figure(figsize=(10, 4))
bars = plt.bar(func_counts.index, func_counts.values)

plt.xlabel("Phrase function")
plt.ylabel("Count")
plt.title("Phrase functions that appear with finite qal forms of yrʾ")

# Add count labels
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        str(int(h)),
        ha="center",
        va="bottom"
    )

# Save high-resolution outputs
plt.savefig("fig03_jara_qal_finite_phrase_functions_noPred.pdf", bbox_inches="tight")
plt.savefig("fig03_jara_qal_finite_phrase_functions_noPred.png", dpi=600, bbox_inches="tight")

plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_25_0.png)
    



```python
FearValData_QalFinite["phrase type Cmpl"].value_counts()

```




    phrase type Cmpl
    PP      271
    VP      248
    CP      125
    NegP    121
    NP       91
    AdvP     24
    PrNP     10
    PPrP     10
    InrP      7
    InjP      2
    IPrP      2
    Name: count, dtype: int64




```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# Styling (as before)
mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# Get counts and exclude 'Pred'
func_counts = (
    FearValData_QalFinite["phrase type Cmpl"]
    .value_counts()
    .drop("VP", errors="ignore")
)

# Plot
plt.figure(figsize=(10, 4))
bars = plt.bar(func_counts.index, func_counts.values)

plt.xlabel("Phrase type")
plt.ylabel("Count")
plt.title("Phrase types that appear with finite qal forms of yrʾ")

# Add count labels
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        str(int(h)),
        ha="center",
        va="bottom"
    )

# Save high-resolution outputs
plt.savefig("fig04_jara_qal_finite_phrase_types_noPred.pdf", bbox_inches="tight")
plt.savefig("fig04_jara_qal_finite_phrase_types_noPred.png", dpi=600, bbox_inches="tight")

plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_27_0.png)
    



```python
import pandas as pd

# 1) keep only Objc / Cmpl phrases
df_oc = FearValData_QalFinite[
    FearValData_QalFinite["function Cmpl"].isin(["Objc", "Cmpl"])
].copy()

# 2) within those phrases, keep only prepositions
df_prep = df_oc[df_oc["word H Cmpl-element-1 part of speech"] == "prep"].copy()

# 3) pick the first prep per phrase (by order) as the "launcher"
df_prep = df_prep.sort_values("order")
prep_launchers = (
    df_prep
    .groupby(["clause #", "phrase # Cmpl"], as_index=False)
    .first()
)

# 4) count launcher prepositions by lexeme_sbl
prep_counts = prep_launchers["lexeme_hebrew_sbl"].value_counts()

prep_counts

```




    lexeme_hebrew_sbl
    yrʾ    100
    Name: count, dtype: int64




```python
# How many Objc/Cmpl phrases total?
n_phrases_total = df_oc[["clause #", "phrase # Pred"]].drop_duplicates().shape[0]

# How many of those have at least one prep inside?
n_phrases_with_prep = prep_launchers.shape[0]

n_phrases_total, n_phrases_with_prep

```




    (116, 100)




```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

plt.figure(figsize=(7, 4))
bars = plt.bar(prep_counts.index.astype(str), prep_counts.values)

plt.xlabel("Preposition (lexeme_sbl)")
plt.ylabel("Count")
plt.title("Prepositions launching Objc/Cmpl phrases (finite qal forms of yrʾ)")

# add counts on top
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        h,
        str(int(h)),
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=0, ha="right")
plt.savefig("fig05_jara_qal_finite_prep_launchers_objc_cmpl.pdf", bbox_inches="tight")
plt.savefig("fig05_jara_qal_finite_prep_launchers_objc_cmpl.png", dpi=600, bbox_inches="tight")
plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_30_0.png)
    



```python
# Count launcher prepositions by book and lexeme_sbl
prep_by_book = (
    prep_launchers
    .groupby(["book", "lexeme_hebrew_sbl"])
    .size()
    .unstack(fill_value=0)
)

prep_by_book

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>lexeme_hebrew_sbl</th>
      <th>yrʾ</th>
    </tr>
    <tr>
      <th>book</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chronica_I</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Deuteronomium</th>
      <td>12</td>
    </tr>
    <tr>
      <th>Ecclesiastes</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Exodus</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Ezechiel</th>
      <td>4</td>
    </tr>
    <tr>
      <th>Haggai</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Hosea</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Iob</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Jeremia</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Jesaia</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Jona</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Josua</th>
      <td>7</td>
    </tr>
    <tr>
      <th>Judices</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Leviticus</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Micha</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Nehemia</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Numeri</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Proverbia</th>
      <td>5</td>
    </tr>
    <tr>
      <th>Psalmi</th>
      <td>10</td>
    </tr>
    <tr>
      <th>Reges_I</th>
      <td>3</td>
    </tr>
    <tr>
      <th>Reges_II</th>
      <td>8</td>
    </tr>
    <tr>
      <th>Samuel_I</th>
      <td>9</td>
    </tr>
    <tr>
      <th>Samuel_II</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Zephania</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
prep_by_book_pct = prep_by_book.div(prep_by_book.sum(axis=1), axis=0) * 100
prep_by_book_pct.round(1)

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>lexeme_hebrew_sbl</th>
      <th>yrʾ</th>
    </tr>
    <tr>
      <th>book</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Chronica_I</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Deuteronomium</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Ecclesiastes</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Exodus</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Ezechiel</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Haggai</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Hosea</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Iob</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Jeremia</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Jesaia</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Jona</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Josua</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Judices</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Leviticus</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Micha</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Nehemia</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Numeri</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Proverbia</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Psalmi</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Reges_I</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Reges_II</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Samuel_I</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Samuel_II</th>
      <td>100.0</td>
    </tr>
    <tr>
      <th>Zephania</th>
      <td>100.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

fig, ax = plt.subplots(figsize=(9, 5))

# Plot stacked bars
prep_by_book.plot(
    kind="bar",
    stacked=True,
    ax=ax
)

ax.set_xlabel("Book")
ax.set_ylabel("Count")
ax.set_title("Prepositions launching Objc/Cmpl phrases with finite qal forms of jara")

# ---- ADD COUNT LABELS ----
for container in ax.containers:
    for bar in container:
        height = bar.get_height()
        if height > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_y() + height / 2,
                f"{int(height)}",
                ha="center",
                va="center",
                fontsize=9
            )

ax.legend(
    title="Preposition (lexeme_hebrew_sbl)",
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)

plt.tight_layout()

plt.savefig(
    "fig06_jara_qal_finite_prep_launchers_by_book.pdf",
    bbox_inches="tight"
)
plt.savefig(
    "fig06_jara_qal_finite_prep_launchers_by_book.png",
    dpi=600,
    bbox_inches="tight"
)
plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_33_0.png)
    


## Jara + MN


```python
FearValenceMN1 = '''
verse verse book chapter
  clause
    p1:phrase function typ
      word lex=JR>[ vs=qal vt=perf|impf|wayq|impv lex_utf8 gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
    p2:phrase function typ=PP
      w1:word lex=MN lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
      w2:word lex lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex


p2 =: w1

'''
FearValenceMN1  = BHS.search(FearValenceMN1)
BHS.table(FearValenceMN1, start=1, end=3, condensed=False, extraFeatures='qere_utf8', fmt='text-orig-full-ketiv', colorMap={3:'yellow', 4:'magenta'})
```

      2.03s 127 results



<table><tr class="tf rtl"><th class="tf rtl">n</th><th class="tf rtl">p</th><th class="tf rtl">verse</th><th class="tf">clause</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">word</th></tr>
<tr class="tf rtl"><td class="tf rtl">1</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=9&amp;verse=30&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 9:30">Exodus 9:30</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">כִּ֚י </span><span class="txtu hbo">טֶ֣רֶם </span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span><span class="txtu hbo">פְּנֵ֖י </span><span class="txtu hbo">יְהוָ֥ה </span><span class="txtu hbo">אֱלֹהִֽים׃ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span><span class="txtu hbo">פְּנֵ֖י </span><span class="txtu hbo">יְהוָ֥ה </span><span class="txtu hbo">אֱלֹהִֽים׃ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">2</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=9&amp;verse=30&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 9:30">Exodus 9:30</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">כִּ֚י </span><span class="txtu hbo">טֶ֣רֶם </span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span><span class="plain rtl  hl" ><span class="txtu hbo">פְּנֵ֖י </span></span><span class="txtu hbo">יְהוָ֥ה </span><span class="txtu hbo">אֱלֹהִֽים׃ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span><span class="plain rtl  hl" ><span class="txtu hbo">פְּנֵ֖י </span></span><span class="txtu hbo">יְהוָ֥ה </span><span class="txtu hbo">אֱלֹהִֽים׃ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">פְּנֵ֖י </span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">3</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=9&amp;verse=30&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 9:30">Exodus 9:30</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">כִּ֚י </span><span class="txtu hbo">טֶ֣רֶם </span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span><span class="txtu hbo">פְּנֵ֖י </span><span class="plain rtl  hl" ><span class="txtu hbo">יְהוָ֥ה </span></span><span class="txtu hbo">אֱלֹהִֽים׃ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירְא֔וּן </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span><span class="txtu hbo">פְּנֵ֖י </span><span class="plain rtl  hl" ><span class="txtu hbo">יְהוָ֥ה </span></span><span class="txtu hbo">אֱלֹהִֽים׃ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">מִ</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">יְהוָ֥ה </span></span></span></td></tr></table>



```python
BHS.export(FearValenceMN1+FearValenceMN1, toDir='/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export', toFile='FearValDataExport_02_MN.tsv')
```


```python
FearValenceMN=pd.read_csv('/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValDataExport_02_MN.tsv',delimiter='\t',encoding='utf-16')
pd.set_option('display.max_columns', 100)
FearValenceMN.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>NODE1</th>
      <th>TYPE1</th>
      <th>TEXT1</th>
      <th>book1</th>
      <th>chapter1</th>
      <th>verse1</th>
      <th>NODE2</th>
      <th>TYPE2</th>
      <th>TEXT2</th>
      <th>NODE3</th>
      <th>TYPE3</th>
      <th>TEXT3</th>
      <th>function3</th>
      <th>typ3</th>
      <th>NODE4</th>
      <th>TYPE4</th>
      <th>TEXT4</th>
      <th>g_cons4</th>
      <th>g_lex4</th>
      <th>g_word4</th>
      <th>gn4</th>
      <th>lex4</th>
      <th>lex_utf84</th>
      <th>macula_BH_lxx_lex_14</th>
      <th>macula_BH_lxx_lex_24</th>
      <th>macula_BH_lxx_lex_34</th>
      <th>macula_BH_lxx_text4</th>
      <th>nu4</th>
      <th>ps4</th>
      <th>sp4</th>
      <th>tov_BH_text4</th>
      <th>tov_LXX_lex4</th>
      <th>tov_LXX_text4</th>
      <th>vs4</th>
      <th>vt4</th>
      <th>NODE5</th>
      <th>TYPE5</th>
      <th>TEXT5</th>
      <th>function5</th>
      <th>typ5</th>
      <th>NODE6</th>
      <th>TYPE6</th>
      <th>TEXT6</th>
      <th>g_cons6</th>
      <th>g_lex6</th>
      <th>g_word6</th>
      <th>gn6</th>
      <th>lex6</th>
      <th>lex_utf86</th>
      <th>macula_BH_lxx_lex_16</th>
      <th>macula_BH_lxx_lex_26</th>
      <th>macula_BH_lxx_lex_36</th>
      <th>macula_BH_lxx_text6</th>
      <th>nu6</th>
      <th>ps6</th>
      <th>sp6</th>
      <th>tov_BH_text6</th>
      <th>tov_LXX_lex6</th>
      <th>tov_LXX_text6</th>
      <th>vs6</th>
      <th>vt6</th>
      <th>NODE7</th>
      <th>TYPE7</th>
      <th>TEXT7</th>
      <th>g_cons7</th>
      <th>g_lex7</th>
      <th>g_word7</th>
      <th>gn7</th>
      <th>lex7</th>
      <th>lex_utf87</th>
      <th>macula_BH_lxx_lex_17</th>
      <th>macula_BH_lxx_lex_27</th>
      <th>macula_BH_lxx_lex_37</th>
      <th>macula_BH_lxx_text7</th>
      <th>nu7</th>
      <th>ps7</th>
      <th>sp7</th>
      <th>tov_BH_text7</th>
      <th>tov_LXX_lex7</th>
      <th>tov_LXX_text7</th>
      <th>vs7</th>
      <th>vt7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>1416161</td>
      <td>verse</td>
      <td>וְאַתָּ֖ה וַעֲבָדֶ֑יךָ יָדַ֕עְתִּי כִּ֚י טֶ֣רֶ...</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>434552</td>
      <td>clause</td>
      <td>כִּ֚י טֶ֣רֶם תִּֽירְא֔וּן מִפְּנֵ֖י יְהוָ֥ה אֱ...</td>
      <td>672561</td>
      <td>phrase</td>
      <td>תִּֽירְא֔וּן</td>
      <td>Pred</td>
      <td>VP</td>
      <td>33688</td>
      <td>word</td>
      <td>תִּֽירְא֔וּן</td>
      <td>TJR&gt;WN</td>
      <td>JR:&gt;</td>
      <td>T.I45JR:&gt;80W.N</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>πεφόβησθε</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>תיראון</td>
      <td>φοβέω</td>
      <td>πεφόβησθε</td>
      <td>qal</td>
      <td>impf</td>
      <td>672562</td>
      <td>phrase</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>33689</td>
      <td>word</td>
      <td>מִ</td>
      <td>M</td>
      <td>MI-</td>
      <td>MI-</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>מ</td>
      <td>NaN</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33689</td>
      <td>word</td>
      <td>מִ</td>
      <td>M</td>
      <td>MI-</td>
      <td>MI-</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>מ</td>
      <td>NaN</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>1416161</td>
      <td>verse</td>
      <td>וְאַתָּ֖ה וַעֲבָדֶ֑יךָ יָדַ֕עְתִּי כִּ֚י טֶ֣רֶ...</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>434552</td>
      <td>clause</td>
      <td>כִּ֚י טֶ֣רֶם תִּֽירְא֔וּן מִפְּנֵ֖י יְהוָ֥ה אֱ...</td>
      <td>672561</td>
      <td>phrase</td>
      <td>תִּֽירְא֔וּן</td>
      <td>Pred</td>
      <td>VP</td>
      <td>33688</td>
      <td>word</td>
      <td>תִּֽירְא֔וּן</td>
      <td>TJR&gt;WN</td>
      <td>JR:&gt;</td>
      <td>T.I45JR:&gt;80W.N</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>πεφόβησθε</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>תיראון</td>
      <td>φοβέω</td>
      <td>πεφόβησθε</td>
      <td>qal</td>
      <td>impf</td>
      <td>672562</td>
      <td>phrase</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>33689</td>
      <td>word</td>
      <td>מִ</td>
      <td>M</td>
      <td>MI-</td>
      <td>MI-</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>מ</td>
      <td>NaN</td>
      <td>-</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>33690</td>
      <td>word</td>
      <td>פְּנֵ֖י</td>
      <td>PNJ</td>
      <td>P.:N</td>
      <td>P.:N;73J</td>
      <td>m</td>
      <td>PNH/</td>
      <td>פנה</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>pl</td>
      <td>NaN</td>
      <td>subs</td>
      <td>פני</td>
      <td>NaN</td>
      <td>?</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(FearValenceMN.columns)
```

    Index(['R', 'S1', 'S2', 'S3', 'NODE1', 'TYPE1', 'TEXT1', 'book1', 'chapter1',
           'verse1', 'NODE2', 'TYPE2', 'TEXT2', 'NODE3', 'TYPE3', 'TEXT3',
           'function3', 'typ3', 'NODE4', 'TYPE4', 'TEXT4', 'g_cons4', 'g_lex4',
           'g_word4', 'gn4', 'lex4', 'lex_utf84', 'macula_BH_lxx_lex_14',
           'macula_BH_lxx_lex_24', 'macula_BH_lxx_lex_34', 'macula_BH_lxx_text4',
           'nu4', 'ps4', 'sp4', 'tov_BH_text4', 'tov_LXX_lex4', 'tov_LXX_text4',
           'vs4', 'vt4', 'NODE5', 'TYPE5', 'TEXT5', 'function5', 'typ5', 'NODE6',
           'TYPE6', 'TEXT6', 'g_cons6', 'g_lex6', 'g_word6', 'gn6', 'lex6',
           'lex_utf86', 'macula_BH_lxx_lex_16', 'macula_BH_lxx_lex_26',
           'macula_BH_lxx_lex_36', 'macula_BH_lxx_text6', 'nu6', 'ps6', 'sp6',
           'tov_BH_text6', 'tov_LXX_lex6', 'tov_LXX_text6', 'vs6', 'vt6', 'NODE7',
           'TYPE7', 'TEXT7', 'g_cons7', 'g_lex7', 'g_word7', 'gn7', 'lex7',
           'lex_utf87', 'macula_BH_lxx_lex_17', 'macula_BH_lxx_lex_27',
           'macula_BH_lxx_lex_37', 'macula_BH_lxx_text7', 'nu7', 'ps7', 'sp7',
           'tov_BH_text7', 'tov_LXX_lex7', 'tov_LXX_text7', 'vs7', 'vt7'],
          dtype='object')



```python
# Rename columns
FearValenceMN = FearValenceMN.rename(columns={

"R":"order",
"book1":"book",
"chapter1":"chapter",
"verse1":"verse",
"NODE2":"clause #",
"TEXT2":"verse_text",



"NODE3":"phrase # Pred",
"function3":"function Pred",
"typ3":"phrase type Pred",
"TEXT3":"phrase text Pred",

"NODE4":"word H # Pred",
"TEXT4":"word H text Pred",
"sp4":"word H Pred part of speech",
"nu4":"word H Pred number",
"ps4":"word H Pred person",
"gn4":"word H Pred gender",
"vs4":"word H Pred stem",
"vt4":"word H Pred tense",
"lex4":"word H lex Pred",
"lex_utf84":"word H lexUTF8 Pred",
"tov_LXX_text4":"word G text Pred",
"tov_LXX_lex4":"word G lex Pred",



"NODE5":"phrase # Cmpl",
"function5":"function Cmpl",
"typ5":"phrase type Cmpl",
"TEXT5":"phrase text Cmpl",

"NODE6":"word H # Cmpl-element-1",
"TEXT6":"word H text Cmpl-element-1",
"sp6":"word H Cmpl-element-1 part of speech",
"nu6":"word H Cmpl-element-1 number",
"ps6":"word H Cmpl-element-1 person",
"gn6":"word H Cmpl-element-1 gender",
"vs6":"word H Cmpl-element-1 stem",
"vt6":"word H Cmpl-element-1 tense",
"lex6":"word H lex Cmpl-element-1",
"lex_utf86":"word H lexUTF8 Cmpl-element-1",
"tov_LXX_text6":"word G text Cmpl-element-1",
"tov_LXX_lex6":"word G lex Cmpl-element-1",

"NODE7":"word H # Cmpl-element-2",
"TEXT7":"word H text Cmpl-element-2",
"sp7":"word H Cmpl-element-2 part of speech",
"nu7":"word H Cmpl-element-2 number",
"ps7":"word H Cmpl-element-2 person",
"gn7":"word H Cmpl-element-2 gender",
"vs7":"word H Cmpl-element-2 stem",
"vt7":"word H Cmpl-element-2 tense",
"lex7":"word H lex Cmpl-element-2",
"lex_utf87":"word H lexUTF8 Cmpl-element-2",
"tov_LXX_text7":"word G text Cmpl-element-2",
"tov_LXX_lex7":"word G lex Cmpl-element-2",


})

# Reduce DataFrame to the selected columns in the desired order
FearValenceMN = FearValenceMN[
[

"order",
"book",
"chapter",
"verse",
"clause #",
"verse_text",



"phrase # Pred",
"function Pred",
"phrase type Pred",
"phrase text Pred",

"word H # Pred",
"word H text Pred",
"word H Pred part of speech",
"word H Pred number",
"word H Pred person",
"word H Pred gender",
"word H Pred stem",
"word H Pred tense",
"word H lex Pred",
"word H lexUTF8 Pred",
"word G text Pred",
"word G lex Pred",



"phrase # Cmpl",
"function Cmpl",
"phrase type Cmpl",
"phrase text Cmpl",

"word H # Cmpl-element-1",
"word H text Cmpl-element-1",
"word H Cmpl-element-1 part of speech",
"word H Cmpl-element-1 number",
"word H Cmpl-element-1 person",
"word H Cmpl-element-1 gender",
"word H Cmpl-element-1 stem",
"word H Cmpl-element-1 tense",
"word H lex Cmpl-element-1",
"word H lexUTF8 Cmpl-element-1",
"word G text Cmpl-element-1",
"word G lex Cmpl-element-1",

"word H # Cmpl-element-2",
"word H text Cmpl-element-2",
"word H Cmpl-element-2 part of speech",
"word H Cmpl-element-2 number",
"word H Cmpl-element-2 person",
"word H Cmpl-element-2 gender",
"word H Cmpl-element-2 stem",
"word H Cmpl-element-2 tense",
"word H lex Cmpl-element-2",
"word H lexUTF8 Cmpl-element-2",
"word G text Cmpl-element-2",
"word G lex Cmpl-element-2",

]
]
FearValenceMN.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>word H # Cmpl-element-2</th>
      <th>word H text Cmpl-element-2</th>
      <th>word H Cmpl-element-2 part of speech</th>
      <th>word H Cmpl-element-2 number</th>
      <th>word H Cmpl-element-2 person</th>
      <th>word H Cmpl-element-2 gender</th>
      <th>word H Cmpl-element-2 stem</th>
      <th>word H Cmpl-element-2 tense</th>
      <th>word H lex Cmpl-element-2</th>
      <th>word H lexUTF8 Cmpl-element-2</th>
      <th>word G text Cmpl-element-2</th>
      <th>word G lex Cmpl-element-2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>434552</td>
      <td>כִּ֚י טֶ֣רֶם תִּֽירְא֔וּן מִפְּנֵ֖י יְהוָ֥ה אֱ...</td>
      <td>672561</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּֽירְא֔וּן</td>
      <td>33688</td>
      <td>תִּֽירְא֔וּן</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>πεφόβησθε</td>
      <td>φοβέω</td>
      <td>672562</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>33689</td>
      <td>מִ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>-</td>
      <td>NaN</td>
      <td>33689</td>
      <td>מִ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>-</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>434552</td>
      <td>כִּ֚י טֶ֣רֶם תִּֽירְא֔וּן מִפְּנֵ֖י יְהוָ֥ה אֱ...</td>
      <td>672561</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּֽירְא֔וּן</td>
      <td>33688</td>
      <td>תִּֽירְא֔וּן</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>πεφόβησθε</td>
      <td>φοβέω</td>
      <td>672562</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>33689</td>
      <td>מִ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>-</td>
      <td>NaN</td>
      <td>33690</td>
      <td>פְּנֵ֖י</td>
      <td>subs</td>
      <td>pl</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>PNH/</td>
      <td>פנה</td>
      <td>?</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>434552</td>
      <td>כִּ֚י טֶ֣רֶם תִּֽירְא֔וּן מִפְּנֵ֖י יְהוָ֥ה אֱ...</td>
      <td>672561</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּֽירְא֔וּן</td>
      <td>33688</td>
      <td>תִּֽירְא֔וּן</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>πεφόβησθε</td>
      <td>φοβέω</td>
      <td>672562</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>33689</td>
      <td>מִ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>-</td>
      <td>NaN</td>
      <td>33691</td>
      <td>יְהוָ֥ה</td>
      <td>nmpr</td>
      <td>sg</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>τὸν κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>434552</td>
      <td>כִּ֚י טֶ֣רֶם תִּֽירְא֔וּן מִפְּנֵ֖י יְהוָ֥ה אֱ...</td>
      <td>672561</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּֽירְא֔וּן</td>
      <td>33688</td>
      <td>תִּֽירְא֔וּן</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>πεφόβησθε</td>
      <td>φοβέω</td>
      <td>672562</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>33689</td>
      <td>מִ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>-</td>
      <td>NaN</td>
      <td>33692</td>
      <td>אֱלֹהִֽים׃</td>
      <td>subs</td>
      <td>pl</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>-</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>14</td>
      <td>439868</td>
      <td>וְיָרֵ֥אתָ מֵּאֱלֹהֶ֖יךָ</td>
      <td>688998</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֥אתָ</td>
      <td>64020</td>
      <td>יָרֵ֥אתָ</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβηθήσῃ</td>
      <td>φοβέω</td>
      <td>688999</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>מֵּאֱלֹהֶ֖יךָ</td>
      <td>64021</td>
      <td>מֵּ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>τὸν σου</td>
      <td>NaN</td>
      <td>64021</td>
      <td>מֵּ</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MN</td>
      <td>מן</td>
      <td>τὸν σου</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValenceMN.to_excel("/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValResearchData_01_MN.xlsx", index=False)
```


```python
lex7_counts = FearValenceMN["word H lex Cmpl-element-2"].value_counts()
lex7_counts
```




    word H lex Cmpl-element-2
    MN        106
    PNH/       40
    >LHJM/     12
    H          12
    DBR/       10
    L           6
    MLK/        6
    JHWH/       6
    PXD/        4
    KFDJ/       4
    XJH/        2
    CD/         2
    MCPV/       2
    RBBH/       2
    R</         2
    >RY/        2
    CMW<H/      2
    LJLH/       2
    >WT/        2
    MJ          2
    <M/         2
    >CWR/       2
    BBL/        2
    M<RB/       2
    >NWC/       2
    <BD/        2
    CLMH=/      2
    CMW>L/      2
    GT=/        2
    >KJC/       2
    DWD==/      2
    PLCTJ/      2
    >C/         2
    CLG/        2
    Name: count, dtype: int64




```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# Styling
mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

plt.figure(figsize=(7, 4))

bars = plt.bar(lex7_counts.index.astype(str), lex7_counts.values)

plt.xlabel("lex7")
plt.ylabel("Count")
plt.title("Distribution of lex7 values")

# Add count labels
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        str(int(h)),
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=30, ha="right")

# Save outputs
plt.savefig("fig07_lex7_distribution.pdf", bbox_inches="tight")
plt.savefig("fig07_lex7_distribution.png", dpi=600, bbox_inches="tight")

plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_42_0.png)
    


## Jara + >T


```python
FearValenceET1 = '''
verse verse book chapter
  clause
    p1:phrase function typ
      word lex=JR>[ vs=qal vt=perf|impf|wayq|impv lex_utf8 gn nu ps sp g_cons g_lex g_word  g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
    p2:phrase function typ=PP
      w1:word lex=>T lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word  g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
      w2:word lex lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word  g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex


p2 =: w1
w1 <: w2

'''
FearValenceET1  = BHS.search(FearValenceET1)
BHS.table(FearValenceET1, start=1, end=3, condensed=False, extraFeatures='qere_utf8', fmt='text-orig-full-ketiv', colorMap={3:'yellow', 4:'magenta'})
```

      2.19s 36 results



<table><tr class="tf rtl"><th class="tf rtl">n</th><th class="tf rtl">p</th><th class="tf rtl">verse</th><th class="tf">clause</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">word</th></tr>
<tr class="tf rtl"><td class="tf rtl">1</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=1&amp;verse=17&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 1:17">Exodus 1:17</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וַ</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span><span class="txtu hbo">הַֽ</span><span class="txtu hbo">מְיַלְּדֹת֙ </span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">הָ֣</span></span><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">הָ֣</span></span><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">הָ֣</span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">2</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=1&amp;verse=21&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 1:21">Exodus 1:21</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">כִּֽי־</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יָֽרְא֥וּ </span></span></span><span class="txtu hbo">הַֽ</span><span class="txtu hbo">מְיַלְּדֹ֖ת </span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">הָ</span></span><span class="txtu hbo">אֱלֹהִ֑ים </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יָֽרְא֥וּ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יָֽרְא֥וּ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">הָ</span></span><span class="txtu hbo">אֱלֹהִ֑ים </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">הָ</span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">3</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=14&amp;verse=31&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 14:31">Exodus 14:31</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וַ</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יִּֽירְא֥וּ </span></span></span><span class="txtu hbo">הָ</span><span class="txtu hbo">עָ֖ם </span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">יְהוָ֑ה </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יִּֽירְא֥וּ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יִּֽירְא֥וּ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">יְהוָ֑ה </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">יְהוָ֑ה </span></span></span></td></tr></table>



```python
FearValenceET2 = '''
verse verse book chapter
  clause
    p1:phrase function typ
      word lex=JR>[ vs=qal vt=perf|impf|wayq|impv lex_utf8 gn nu ps sp g_cons g_lex g_word g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
    p2:phrase function typ=PP
      w1:word lex=>T lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word  g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
      w2:word lex lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word  g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex


p2 =: w1

'''
FearValenceET2  = BHS.search(FearValenceET2)
BHS.table(FearValenceET2, start=1, end=3, condensed=False, extraFeatures='qere_utf8', fmt='text-orig-full-ketiv', colorMap={3:'yellow', 4:'magenta'})
```

      2.03s 113 results



<table><tr class="tf rtl"><th class="tf rtl">n</th><th class="tf rtl">p</th><th class="tf rtl">verse</th><th class="tf">clause</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">word</th></tr>
<tr class="tf rtl"><td class="tf rtl">1</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=1&amp;verse=17&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 1:17">Exodus 1:17</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וַ</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span><span class="txtu hbo">הַֽ</span><span class="txtu hbo">מְיַלְּדֹת֙ </span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="txtu hbo">הָ֣</span><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="txtu hbo">הָ֣</span><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">2</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=1&amp;verse=17&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 1:17">Exodus 1:17</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וַ</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span><span class="txtu hbo">הַֽ</span><span class="txtu hbo">מְיַלְּדֹת֙ </span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">הָ֣</span></span><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="plain rtl  hl" ><span class="txtu hbo">הָ֣</span></span><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">הָ֣</span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">3</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Exodus&amp;chapter=1&amp;verse=17&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Exodus 1:17">Exodus 1:17</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וַ</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span><span class="txtu hbo">הַֽ</span><span class="txtu hbo">מְיַלְּדֹת֙ </span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="txtu hbo">הָ֣</span><span class="plain rtl  hl" ><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירֶ֤אןָ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span><span class="txtu hbo">הָ֣</span><span class="plain rtl  hl" ><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֶת־</span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אֱלֹהִ֔ים </span></span></span></td></tr></table>



```python
BHS.export(FearValenceET1+FearValenceET2, toDir='/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/', toFile='FearValDataExport_03_ET.tsv')
```


```python
FearValenceET=pd.read_csv('/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValDataExport_03_ET.tsv',delimiter='\t',encoding='utf-16')
pd.set_option('display.max_columns', 100)
FearValenceET.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>NODE1</th>
      <th>TYPE1</th>
      <th>TEXT1</th>
      <th>book1</th>
      <th>chapter1</th>
      <th>verse1</th>
      <th>NODE2</th>
      <th>TYPE2</th>
      <th>TEXT2</th>
      <th>NODE3</th>
      <th>TYPE3</th>
      <th>TEXT3</th>
      <th>function3</th>
      <th>typ3</th>
      <th>NODE4</th>
      <th>TYPE4</th>
      <th>TEXT4</th>
      <th>g_cons4</th>
      <th>g_lex4</th>
      <th>g_word4</th>
      <th>gn4</th>
      <th>lex4</th>
      <th>lex_utf84</th>
      <th>macula_BH_lxx_lex_14</th>
      <th>macula_BH_lxx_lex_24</th>
      <th>macula_BH_lxx_lex_34</th>
      <th>macula_BH_lxx_text4</th>
      <th>nu4</th>
      <th>ps4</th>
      <th>sp4</th>
      <th>tov_BH_text4</th>
      <th>tov_LXX_lex4</th>
      <th>tov_LXX_text4</th>
      <th>vs4</th>
      <th>vt4</th>
      <th>NODE5</th>
      <th>TYPE5</th>
      <th>TEXT5</th>
      <th>function5</th>
      <th>typ5</th>
      <th>NODE6</th>
      <th>TYPE6</th>
      <th>TEXT6</th>
      <th>g_cons6</th>
      <th>g_lex6</th>
      <th>g_word6</th>
      <th>gn6</th>
      <th>lex6</th>
      <th>lex_utf86</th>
      <th>macula_BH_lxx_lex_16</th>
      <th>macula_BH_lxx_lex_26</th>
      <th>macula_BH_lxx_lex_36</th>
      <th>macula_BH_lxx_text6</th>
      <th>nu6</th>
      <th>ps6</th>
      <th>sp6</th>
      <th>tov_BH_text6</th>
      <th>tov_LXX_lex6</th>
      <th>tov_LXX_text6</th>
      <th>vs6</th>
      <th>vt6</th>
      <th>NODE7</th>
      <th>TYPE7</th>
      <th>TEXT7</th>
      <th>g_cons7</th>
      <th>g_lex7</th>
      <th>g_word7</th>
      <th>gn7</th>
      <th>lex7</th>
      <th>lex_utf87</th>
      <th>macula_BH_lxx_lex_17</th>
      <th>macula_BH_lxx_lex_27</th>
      <th>macula_BH_lxx_lex_37</th>
      <th>macula_BH_lxx_text7</th>
      <th>nu7</th>
      <th>ps7</th>
      <th>sp7</th>
      <th>tov_BH_text7</th>
      <th>tov_LXX_lex7</th>
      <th>tov_LXX_text7</th>
      <th>vs7</th>
      <th>vt7</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Exodus</td>
      <td>1</td>
      <td>17</td>
      <td>1415938</td>
      <td>verse</td>
      <td>וַתִּירֶ֤אןָ הַֽמְיַלְּדֹת֙ אֶת־הָ֣אֱלֹהִ֔ים ו...</td>
      <td>Exodus</td>
      <td>1</td>
      <td>17</td>
      <td>433593</td>
      <td>clause</td>
      <td>וַתִּירֶ֤אןָ הַֽמְיַלְּדֹת֙ אֶת־הָ֣אֱלֹהִ֔ים</td>
      <td>669693</td>
      <td>phrase</td>
      <td>תִּירֶ֤אןָ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>29000</td>
      <td>word</td>
      <td>תִּירֶ֤אןָ</td>
      <td>TJR&gt;N</td>
      <td>JRE&gt;</td>
      <td>T.IJRE70&gt;N@</td>
      <td>f</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβήθησαν</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>תיראן</td>
      <td>φοβέω</td>
      <td>ἐφοβήθησαν</td>
      <td>qal</td>
      <td>wayq</td>
      <td>669695</td>
      <td>phrase</td>
      <td>אֶת־הָ֣אֱלֹהִ֔ים</td>
      <td>Objc</td>
      <td>PP</td>
      <td>29003</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29004</td>
      <td>word</td>
      <td>הָ֣</td>
      <td>H</td>
      <td>H@-</td>
      <td>H@74-</td>
      <td>NaN</td>
      <td>H</td>
      <td>ה</td>
      <td>ὁ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>τὸν</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>art</td>
      <td>ה</td>
      <td>ὁ</td>
      <td>τὸν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Exodus</td>
      <td>1</td>
      <td>21</td>
      <td>1415942</td>
      <td>verse</td>
      <td>וַיְהִ֕י כִּֽי־יָֽרְא֥וּ הַֽמְיַלְּדֹ֖ת אֶת־הָ...</td>
      <td>Exodus</td>
      <td>1</td>
      <td>21</td>
      <td>433610</td>
      <td>clause</td>
      <td>כִּֽי־יָֽרְא֥וּ הַֽמְיַלְּדֹ֖ת אֶת־הָאֱלֹהִ֑ים</td>
      <td>669749</td>
      <td>phrase</td>
      <td>יָֽרְא֥וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>29083</td>
      <td>word</td>
      <td>יָֽרְא֥וּ</td>
      <td>JR&gt;W</td>
      <td>J@R:&gt;</td>
      <td>J@45R:&gt;71W.</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβοῦντο</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראו</td>
      <td>φοβέω</td>
      <td>ἐφοβοῦντο</td>
      <td>qal</td>
      <td>perf</td>
      <td>669751</td>
      <td>phrase</td>
      <td>אֶת־הָאֱלֹהִ֑ים</td>
      <td>Objc</td>
      <td>PP</td>
      <td>29086</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29087</td>
      <td>word</td>
      <td>הָ</td>
      <td>H</td>
      <td>H@-</td>
      <td>H@-</td>
      <td>NaN</td>
      <td>H</td>
      <td>ה</td>
      <td>ὁ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>τὸν</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>art</td>
      <td>ה</td>
      <td>ὁ</td>
      <td>τὸν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Exodus</td>
      <td>14</td>
      <td>31</td>
      <td>1416309</td>
      <td>verse</td>
      <td>וַיַּ֨רְא יִשְׂרָאֵ֜ל אֶת־הַיָּ֣ד הַגְּדֹלָ֗ה ...</td>
      <td>Exodus</td>
      <td>14</td>
      <td>31</td>
      <td>435129</td>
      <td>clause</td>
      <td>וַיִּֽירְא֥וּ הָעָ֖ם אֶת־יְהוָ֑ה</td>
      <td>674410</td>
      <td>phrase</td>
      <td>יִּֽירְא֥וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>36902</td>
      <td>word</td>
      <td>יִּֽירְא֥וּ</td>
      <td>JJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>J.I45JR:&gt;71W.</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἐφοβήθη</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>ייראו</td>
      <td>φοβέω</td>
      <td>ἐφοβήθη</td>
      <td>qal</td>
      <td>wayq</td>
      <td>674412</td>
      <td>phrase</td>
      <td>אֶת־יְהוָ֑ה</td>
      <td>Objc</td>
      <td>PP</td>
      <td>36905</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36906</td>
      <td>word</td>
      <td>יְהוָ֑ה</td>
      <td>JHWH</td>
      <td>J:HW@H</td>
      <td>J:HW@92H</td>
      <td>m</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>τὸν κύριον</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Numbers</td>
      <td>14</td>
      <td>9</td>
      <td>1418506</td>
      <td>verse</td>
      <td>אַ֣ךְ בַּֽיהוָה֮ אַל־תִּמְרֹדוּ֒ וְאַתֶּ֗ם אַל...</td>
      <td>Numeri</td>
      <td>14</td>
      <td>9</td>
      <td>442388</td>
      <td>clause</td>
      <td>וְאַתֶּ֗ם אַל־תִּֽירְאוּ֙ אֶת־עַ֣ם הָאָ֔רֶץ</td>
      <td>696605</td>
      <td>phrase</td>
      <td>תִּֽירְאוּ֙</td>
      <td>Pred</td>
      <td>VP</td>
      <td>78581</td>
      <td>word</td>
      <td>תִּֽירְאוּ֙</td>
      <td>TJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>T.I45JR:&gt;W.03</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβηθῆτε</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>תיראו</td>
      <td>φοβέω</td>
      <td>φοβηθῆτε</td>
      <td>qal</td>
      <td>impf</td>
      <td>696606</td>
      <td>phrase</td>
      <td>אֶת־עַ֣ם הָאָ֔רֶץ</td>
      <td>Objc</td>
      <td>PP</td>
      <td>78582</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78583</td>
      <td>word</td>
      <td>עַ֣ם</td>
      <td>&lt;M</td>
      <td>&lt;AM</td>
      <td>&lt;A74M</td>
      <td>m</td>
      <td>&lt;M/</td>
      <td>עם</td>
      <td>λαός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>λαὸν</td>
      <td>sg</td>
      <td>NaN</td>
      <td>subs</td>
      <td>עם</td>
      <td>λαός</td>
      <td>τὸν λαὸν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Deuteronomy</td>
      <td>6</td>
      <td>2</td>
      <td>1419478</td>
      <td>verse</td>
      <td>לְמַ֨עַן תִּירָ֜א אֶת־יְהוָ֣ה אֱלֹהֶ֗יךָ לִ֠שׁ...</td>
      <td>Deuteronomium</td>
      <td>6</td>
      <td>2</td>
      <td>445647</td>
      <td>clause</td>
      <td>לְמַ֨עַן תִּירָ֜א אֶת־יְהוָ֣ה אֱלֹהֶ֗יךָ אַתָּ...</td>
      <td>706500</td>
      <td>phrase</td>
      <td>תִּירָ֜א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>96992</td>
      <td>word</td>
      <td>תִּירָ֜א</td>
      <td>TJR&gt;</td>
      <td>JR@&gt;</td>
      <td>T.IJR@61&gt;</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβῆσθε</td>
      <td>sg</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא</td>
      <td>φοβέω</td>
      <td>φοβῆσθε</td>
      <td>qal</td>
      <td>impf</td>
      <td>706501</td>
      <td>phrase</td>
      <td>אֶת־יְהוָ֣ה אֱלֹהֶ֗יךָ</td>
      <td>Objc</td>
      <td>PP</td>
      <td>96993</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96994</td>
      <td>word</td>
      <td>יְהוָ֣ה</td>
      <td>JHWH</td>
      <td>J:HW@H</td>
      <td>J:HW@74H</td>
      <td>m</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>κύριον</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>κύριον</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Deuteronomy</td>
      <td>6</td>
      <td>13</td>
      <td>1419489</td>
      <td>verse</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ תִּירָ֖א וְאֹתֹ֣ו תַעֲב...</td>
      <td>Deuteronomium</td>
      <td>6</td>
      <td>13</td>
      <td>445689</td>
      <td>clause</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ תִּירָ֖א</td>
      <td>706614</td>
      <td>phrase</td>
      <td>תִּירָ֖א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>97194</td>
      <td>word</td>
      <td>תִּירָ֖א</td>
      <td>TJR&gt;</td>
      <td>JR@&gt;</td>
      <td>T.IJR@73&gt;</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβηθήσῃ</td>
      <td>sg</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא</td>
      <td>φοβέω</td>
      <td>φοβηθήσῃ</td>
      <td>qal</td>
      <td>impf</td>
      <td>706613</td>
      <td>phrase</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ</td>
      <td>Objc</td>
      <td>PP</td>
      <td>97191</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>97192</td>
      <td>word</td>
      <td>יְהוָ֧ה</td>
      <td>JHWH</td>
      <td>J:HW@H</td>
      <td>J:HW@94H</td>
      <td>m</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>κύριον</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>κύριον</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Deuteronomy</td>
      <td>10</td>
      <td>20</td>
      <td>1419596</td>
      <td>verse</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ תִּירָ֖א אֹתֹ֣ו תַעֲבֹ֑...</td>
      <td>Deuteronomium</td>
      <td>10</td>
      <td>20</td>
      <td>446142</td>
      <td>clause</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ תִּירָ֖א</td>
      <td>707957</td>
      <td>phrase</td>
      <td>תִּירָ֖א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>99544</td>
      <td>word</td>
      <td>תִּירָ֖א</td>
      <td>TJR&gt;</td>
      <td>JR@&gt;</td>
      <td>T.IJR@73&gt;</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβηθήσῃ</td>
      <td>sg</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא</td>
      <td>φοβέω</td>
      <td>φοβηθήσῃ</td>
      <td>qal</td>
      <td>impf</td>
      <td>707956</td>
      <td>phrase</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ</td>
      <td>Objc</td>
      <td>PP</td>
      <td>99541</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>99542</td>
      <td>word</td>
      <td>יְהוָ֧ה</td>
      <td>JHWH</td>
      <td>J:HW@H</td>
      <td>J:HW@94H</td>
      <td>m</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>κύριον</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>κύριον</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Deuteronomy</td>
      <td>31</td>
      <td>12</td>
      <td>1420130</td>
      <td>verse</td>
      <td>הַקְהֵ֣ל אֶת־הָעָ֗ם הָֽאֲנָשִׁ֤ים וְהַנָּשִׁים...</td>
      <td>Deuteronomium</td>
      <td>31</td>
      <td>12</td>
      <td>448294</td>
      <td>clause</td>
      <td>וְיָֽרְאוּ֙ אֶת־יְהוָ֣ה אֱלֹֽהֵיכֶ֔ם</td>
      <td>714529</td>
      <td>phrase</td>
      <td>יָֽרְאוּ֙</td>
      <td>Pred</td>
      <td>VP</td>
      <td>110943</td>
      <td>word</td>
      <td>יָֽרְאוּ֙</td>
      <td>JR&gt;W</td>
      <td>J@R:&gt;</td>
      <td>J@45R:&gt;W.03</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβεῖσθαι</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראו</td>
      <td>φοβέω</td>
      <td>φοβεῖσθαι</td>
      <td>qal</td>
      <td>perf</td>
      <td>714530</td>
      <td>phrase</td>
      <td>אֶת־יְהוָ֣ה אֱלֹֽהֵיכֶ֔ם</td>
      <td>Objc</td>
      <td>PP</td>
      <td>110944</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>110945</td>
      <td>word</td>
      <td>יְהוָ֣ה</td>
      <td>JHWH</td>
      <td>J:HW@H</td>
      <td>J:HW@74H</td>
      <td>m</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>κύριον</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>κύριον</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Joshua</td>
      <td>4</td>
      <td>14</td>
      <td>1420314</td>
      <td>verse</td>
      <td>בַּיֹּ֣ום הַה֗וּא גִּדַּ֤ל יְהוָה֙ אֶת־יְהֹושׁ...</td>
      <td>Josua</td>
      <td>4</td>
      <td>14</td>
      <td>449081</td>
      <td>clause</td>
      <td>כַּאֲשֶׁ֛ר יָרְא֥וּ אֶת־מֹשֶׁ֖ה כָּל־יְמֵ֥י חַ...</td>
      <td>716783</td>
      <td>phrase</td>
      <td>יָרְא֥וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>114739</td>
      <td>word</td>
      <td>יָרְא֥וּ</td>
      <td>JR&gt;W</td>
      <td>J@R:&gt;</td>
      <td>J@R:&gt;71W.</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראו</td>
      <td>NaN</td>
      <td>-</td>
      <td>qal</td>
      <td>perf</td>
      <td>716784</td>
      <td>phrase</td>
      <td>אֶת־מֹשֶׁ֖ה</td>
      <td>Objc</td>
      <td>PP</td>
      <td>114740</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>114741</td>
      <td>word</td>
      <td>מֹשֶׁ֖ה</td>
      <td>MCH</td>
      <td>MOCEH</td>
      <td>MOCE73H</td>
      <td>m</td>
      <td>MCH=/</td>
      <td>משׁה</td>
      <td>Μωσεύς</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>μωυσῆν</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>משׁה</td>
      <td>Μωσεύς</td>
      <td>μωυσῆν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Joshua</td>
      <td>4</td>
      <td>24</td>
      <td>1420324</td>
      <td>verse</td>
      <td>לְ֠מַעַן דַּ֜עַת כָּל־עַמֵּ֤י הָאָ֨רֶץ֙ אֶת־יַ...</td>
      <td>Josua</td>
      <td>4</td>
      <td>24</td>
      <td>449113</td>
      <td>clause</td>
      <td>לְמַ֧עַן יְרָאתֶ֛ם אֶת־יְהוָ֥ה אֱלֹהֵיכֶ֖ם כָּ...</td>
      <td>716876</td>
      <td>phrase</td>
      <td>יְרָאתֶ֛ם</td>
      <td>Pred</td>
      <td>VP</td>
      <td>114931</td>
      <td>word</td>
      <td>יְרָאתֶ֛ם</td>
      <td>JR&gt;TM</td>
      <td>J:R@&gt;</td>
      <td>J:R@&gt;TE91M</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>יראתם</td>
      <td>ὑμεῖς σέβομαι</td>
      <td>ὑμεῖς σέβησθε</td>
      <td>qal</td>
      <td>perf</td>
      <td>716877</td>
      <td>phrase</td>
      <td>אֶת־יְהוָ֥ה אֱלֹהֵיכֶ֖ם</td>
      <td>Objc</td>
      <td>PP</td>
      <td>114932</td>
      <td>word</td>
      <td>אֶת־</td>
      <td>&gt;T</td>
      <td>&gt;ET</td>
      <td>&gt;ET</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>prep</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>114933</td>
      <td>word</td>
      <td>יְהוָ֥ה</td>
      <td>JHWH</td>
      <td>J:HW@H</td>
      <td>J:HW@71H</td>
      <td>m</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>κύριον</td>
      <td>sg</td>
      <td>NaN</td>
      <td>nmpr</td>
      <td>יהוה</td>
      <td>κύριος</td>
      <td>κύριον</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename columns
FearValenceET = FearValenceET.rename(columns={

"R":"order",
"book1":"book",
"chapter1":"chapter",
"verse1":"verse",
"NODE2":"clause #",
"TEXT2":"verse_text",



"NODE3":"phrase # Pred",
"function3":"function Pred",
"typ3":"phrase type Pred",
"TEXT3":"phrase text Pred",

"NODE4":"word H # Pred",
"TEXT4":"word H text Pred",
"sp4":"word H Pred part of speech",
"nu4":"word H Pred number",
"ps4":"word H Pred person",
"gn4":"word H Pred gender",
"vs4":"word H Pred stem",
"vt4":"word H Pred tense",
"lex4":"word H lex Pred",
"lex_utf84":"word H lexUTF8 Pred",
"tov_LXX_text4":"word G text Pred",
"tov_LXX_lex4":"word G lex Pred",



"NODE5":"phrase # Cmpl",
"function5":"function Cmpl",
"typ5":"phrase type Cmpl",
"TEXT5":"phrase text Cmpl",

"NODE6":"word H # Cmpl-element-1",
"TEXT6":"word H text Cmpl-element-1",
"sp6":"word H Cmpl-element-1 part of speech",
"nu6":"word H Cmpl-element-1 number",
"ps6":"word H Cmpl-element-1 person",
"gn6":"word H Cmpl-element-1 gender",
"vs6":"word H Cmpl-element-1 stem",
"vt6":"word H Cmpl-element-1 tense",
"lex6":"word H lex Cmpl-element-1",
"lex_utf86":"word H lexUTF8 Cmpl-element-1",
"tov_LXX_text6":"word G text Cmpl-element-1",
"tov_LXX_lex6":"word G lex Cmpl-element-1",

"NODE7":"word H # Cmpl-element-2",
"TEXT7":"word H text Cmpl-element-2",
"sp7":"word H Cmpl-element-2 part of speech",
"nu7":"word H Cmpl-element-2 number",
"ps7":"word H Cmpl-element-2 person",
"gn7":"word H Cmpl-element-2 gender",
"vs7":"word H Cmpl-element-2 stem",
"vt7":"word H Cmpl-element-2 tense",
"lex7":"word H lex Cmpl-element-2",
"lex_utf87":"word H lexUTF8 Cmpl-element-2",
"tov_LXX_text7":"word G text Cmpl-element-2",
"tov_LXX_lex7":"word G lex Cmpl-element-2",


})

# Reduce DataFrame to the selected columns in the desired order
FearValenceET = FearValenceET[
[

"order",
"book",
"chapter",
"verse",
"clause #",
"verse_text",



"phrase # Pred",
"function Pred",
"phrase type Pred",
"phrase text Pred",

"word H # Pred",
"word H text Pred",
"word H Pred part of speech",
"word H Pred number",
"word H Pred person",
"word H Pred gender",
"word H Pred stem",
"word H Pred tense",
"word H lex Pred",
"word H lexUTF8 Pred",
"word G text Pred",
"word G lex Pred",



"phrase # Cmpl",
"function Cmpl",
"phrase type Cmpl",
"phrase text Cmpl",

"word H # Cmpl-element-1",
"word H text Cmpl-element-1",
"word H Cmpl-element-1 part of speech",
"word H Cmpl-element-1 number",
"word H Cmpl-element-1 person",
"word H Cmpl-element-1 gender",
"word H Cmpl-element-1 stem",
"word H Cmpl-element-1 tense",
"word H lex Cmpl-element-1",
"word H lexUTF8 Cmpl-element-1",
"word G text Cmpl-element-1",
"word G lex Cmpl-element-1",

"word H # Cmpl-element-2",
"word H text Cmpl-element-2",
"word H Cmpl-element-2 part of speech",
"word H Cmpl-element-2 number",
"word H Cmpl-element-2 person",
"word H Cmpl-element-2 gender",
"word H Cmpl-element-2 stem",
"word H Cmpl-element-2 tense",
"word H lex Cmpl-element-2",
"word H lexUTF8 Cmpl-element-2",
"word G text Cmpl-element-2",
"word G lex Cmpl-element-2",

]
]
FearValenceET.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-1</th>
      <th>word H text Cmpl-element-1</th>
      <th>word H Cmpl-element-1 part of speech</th>
      <th>word H Cmpl-element-1 number</th>
      <th>word H Cmpl-element-1 person</th>
      <th>word H Cmpl-element-1 gender</th>
      <th>word H Cmpl-element-1 stem</th>
      <th>word H Cmpl-element-1 tense</th>
      <th>word H lex Cmpl-element-1</th>
      <th>word H lexUTF8 Cmpl-element-1</th>
      <th>word G text Cmpl-element-1</th>
      <th>word G lex Cmpl-element-1</th>
      <th>word H # Cmpl-element-2</th>
      <th>word H text Cmpl-element-2</th>
      <th>word H Cmpl-element-2 part of speech</th>
      <th>word H Cmpl-element-2 number</th>
      <th>word H Cmpl-element-2 person</th>
      <th>word H Cmpl-element-2 gender</th>
      <th>word H Cmpl-element-2 stem</th>
      <th>word H Cmpl-element-2 tense</th>
      <th>word H lex Cmpl-element-2</th>
      <th>word H lexUTF8 Cmpl-element-2</th>
      <th>word G text Cmpl-element-2</th>
      <th>word G lex Cmpl-element-2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Exodus</td>
      <td>1</td>
      <td>17</td>
      <td>433593</td>
      <td>וַתִּירֶ֤אןָ הַֽמְיַלְּדֹת֙ אֶת־הָ֣אֱלֹהִ֔ים</td>
      <td>669693</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירֶ֤אןָ</td>
      <td>29000</td>
      <td>תִּירֶ֤אןָ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>f</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθησαν</td>
      <td>φοβέω</td>
      <td>669695</td>
      <td>Objc</td>
      <td>PP</td>
      <td>אֶת־הָ֣אֱלֹהִ֔ים</td>
      <td>29003</td>
      <td>אֶת־</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29004</td>
      <td>הָ֣</td>
      <td>art</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>H</td>
      <td>ה</td>
      <td>τὸν</td>
      <td>ὁ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Exodus</td>
      <td>1</td>
      <td>21</td>
      <td>433610</td>
      <td>כִּֽי־יָֽרְא֥וּ הַֽמְיַלְּדֹ֖ת אֶת־הָאֱלֹהִ֑ים</td>
      <td>669749</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָֽרְא֥וּ</td>
      <td>29083</td>
      <td>יָֽרְא֥וּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>unknown</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβοῦντο</td>
      <td>φοβέω</td>
      <td>669751</td>
      <td>Objc</td>
      <td>PP</td>
      <td>אֶת־הָאֱלֹהִ֑ים</td>
      <td>29086</td>
      <td>אֶת־</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>29087</td>
      <td>הָ</td>
      <td>art</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>H</td>
      <td>ה</td>
      <td>τὸν</td>
      <td>ὁ</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Exodus</td>
      <td>14</td>
      <td>31</td>
      <td>435129</td>
      <td>וַיִּֽירְא֥וּ הָעָ֖ם אֶת־יְהוָ֑ה</td>
      <td>674410</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יִּֽירְא֥וּ</td>
      <td>36902</td>
      <td>יִּֽירְא֥וּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>674412</td>
      <td>Objc</td>
      <td>PP</td>
      <td>אֶת־יְהוָ֑ה</td>
      <td>36905</td>
      <td>אֶת־</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>36906</td>
      <td>יְהוָ֑ה</td>
      <td>nmpr</td>
      <td>sg</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>τὸν κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Numeri</td>
      <td>14</td>
      <td>9</td>
      <td>442388</td>
      <td>וְאַתֶּ֗ם אַל־תִּֽירְאוּ֙ אֶת־עַ֣ם הָאָ֔רֶץ</td>
      <td>696605</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּֽירְאוּ֙</td>
      <td>78581</td>
      <td>תִּֽירְאוּ֙</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβηθῆτε</td>
      <td>φοβέω</td>
      <td>696606</td>
      <td>Objc</td>
      <td>PP</td>
      <td>אֶת־עַ֣ם הָאָ֔רֶץ</td>
      <td>78582</td>
      <td>אֶת־</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>78583</td>
      <td>עַ֣ם</td>
      <td>subs</td>
      <td>sg</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&lt;M/</td>
      <td>עם</td>
      <td>τὸν λαὸν</td>
      <td>λαός</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Deuteronomium</td>
      <td>6</td>
      <td>2</td>
      <td>445647</td>
      <td>לְמַ֨עַן תִּירָ֜א אֶת־יְהוָ֣ה אֱלֹהֶ֗יךָ אַתָּ...</td>
      <td>706500</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֜א</td>
      <td>96992</td>
      <td>תִּירָ֜א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβῆσθε</td>
      <td>φοβέω</td>
      <td>706501</td>
      <td>Objc</td>
      <td>PP</td>
      <td>אֶת־יְהוָ֣ה אֱלֹהֶ֗יךָ</td>
      <td>96993</td>
      <td>אֶת־</td>
      <td>prep</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;T</td>
      <td>את</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>96994</td>
      <td>יְהוָ֣ה</td>
      <td>nmpr</td>
      <td>sg</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>JHWH/</td>
      <td>יהוה</td>
      <td>κύριον</td>
      <td>κύριος</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValenceET.to_excel("/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValResearchData_02_DOwiET.xlsx", index=False)
```


```python
lex7_counts = FearValenceET["word G lex Cmpl-element-2"].value_counts()
lex7_counts
```




    word G lex Cmpl-element-2
    κύριος             40
    θεός               16
    ὁ                  15
    ὄνομα               4
    [unmapped:τον]      4
    λαός                3
    καί                 2
    [unmapped:θεον]     2
    φόβος               2
    οἶκος               2
    Μωσεύς              2
    γῆ                  1
    Ἀμορραῖος           1
    ἀνήρ                1
    πόλις               1
    Σαμουήλ             1
    ὅρκος               1
    βασιλεύς            1
    Σαλωμών             1
    [unmapped:ἀπ]       1
    [unmapped:συν]      1
    πατήρ               1
    Name: count, dtype: int64




```python
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Styling
mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

# --- counts ---
mt_counts = FearValenceET["word H lex Cmpl-element-2"].value_counts()
lxx_counts = FearValenceET["word G lex Cmpl-element-2"].value_counts()

combined = (
    mt_counts.rename("MT")
    .to_frame()
    .join(lxx_counts.rename("LXX"), how="outer")
    .fillna(0)
    .astype(int)
)

combined["total"] = combined["MT"] + combined["LXX"]
combined = combined.sort_values("total", ascending=False).drop(columns="total")


# --- build label lookup table ---
# adjust column names here if your Hebrew / Greek columns have different names
label_df = (
    FearValenceET[["word H lex Cmpl-element-2", "word H lexUTF8 Cmpl-element-2", "word G text Cmpl-element-2", "word G lex Cmpl-element-2"]]
    .dropna(subset=["word H lex Cmpl-element-2", "word G lex Cmpl-element-2"])
    .drop_duplicates()
)

# for each Hebrew lexeme (lex7), keep the first matching Greek + transliterations
label_lookup = (
    label_df.groupby("word H lex Cmpl-element-2", as_index=True)
    .first()
)

# --- create display labels ---
labels = []
for heb in combined.index:
    if heb in label_lookup.index:
        heb_sbl = label_lookup.loc[heb, "word G lex Cmpl-element-2"]
        grk = label_lookup.loc[heb, "word G lex Cmpl-element-2"]
       # grk_sbl = label_lookup.loc[heb, "lexeme_greek_sbl"]
       # labels.append(f"{heb} ({heb_sbl}) | {grk} ({grk_sbl})")
        labels.append(f"{heb} | {grk} ")

    else:
        labels.append(str(heb))

# --- plot ---
plt.figure(figsize=(10, 5))

x = np.arange(len(combined.index))
width = 0.4

bars1 = plt.bar(x - width / 2, combined["MT"], width, label="MT")
bars2 = plt.bar(x + width / 2, combined["LXX"], width, label="LXX")

plt.xlabel("Lexeme")
plt.ylabel("Count")
plt.title("Distribution of Lexemes Governed by >T")

# Add count labels
for bars in [bars1, bars2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                h,
                str(int(h)),
                ha="center",
                va="bottom"
            )

plt.xticks(x, labels, rotation=30, ha="right")
plt.legend()

# Save outputs
plt.savefig("fig08_lex7_distribution.pdf", bbox_inches="tight")
plt.savefig("fig08_lex7_distribution.png", dpi=600, bbox_inches="tight")

plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_51_0.png)
    


## Jara + Direct Object wo ET


```python
FearValenceDO1 = '''
verse verse book chapter
  clause
    p1:phrase function=Pred typ
      word lex=JR>[ vs=qal vt=perf|impf|wayq|impv lex_utf8 gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
    p2:phrase function=Objc typ
    /without/
      word lex=>T lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex
    /-/
      w1:word lex lex_utf8 vs vt gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex


p2 =: w1

'''
FearValenceDO1  = BHS.search(FearValenceDO1)
BHS.table(FearValenceDO1, start=1, end=3, condensed=False, extraFeatures='qere_utf8', fmt='text-orig-full-ketiv', colorMap={3:'yellow', 4:'magenta'})
```

      1.69s 18 results



<table><tr class="tf rtl"><th class="tf rtl">n</th><th class="tf rtl">p</th><th class="tf rtl">verse</th><th class="tf">clause</th><th class="tf">phrase</th><th class="tf">word</th><th class="tf">phrase</th><th class="tf">word</th></tr>
<tr class="tf rtl"><td class="tf rtl">1</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Leviticus&amp;chapter=19&amp;verse=3&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Leviticus 19:3">Leviticus 19:3</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אִמֹּ֤ו </span></span><span class="txtu hbo">וְ</span><span class="txtu hbo">אָבִיו֙ </span></span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֔אוּ </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֔אוּ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֔אוּ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">אִמֹּ֤ו </span></span><span class="txtu hbo">וְ</span><span class="txtu hbo">אָבִיו֙ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">אִמֹּ֤ו </span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">2</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Leviticus&amp;chapter=19&amp;verse=30&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Leviticus 19:30">Leviticus 19:30</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וּ</span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִקְדָּשִׁ֖י </span></span></span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֑אוּ </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֑אוּ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֑אוּ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִקְדָּשִׁ֖י </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">מִקְדָּשִׁ֖י </span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">3</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Leviticus&amp;chapter=26&amp;verse=2&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Leviticus 26:2">Leviticus 26:2</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">וּ</span><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִקְדָּשִׁ֖י </span></span></span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֑אוּ </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֑אוּ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָ֑אוּ </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="plain rtl  hl" ><span class="txtu hbo">מִקְדָּשִׁ֖י </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl" ><span class="txtu hbo">מִקְדָּשִׁ֖י </span></span></span></td></tr></table>



```python
BHS.export(FearValenceDO1, toDir='/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/', toFile='FearValDataExport_04_DO_woET.tsv')
```


```python
FearValenceDO1=pd.read_csv('/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValDataExport_04_DO_woET.tsv',delimiter='\t',encoding='utf-16')
pd.set_option('display.max_columns', 50)
FearValenceDO1.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>NODE1</th>
      <th>TYPE1</th>
      <th>TEXT1</th>
      <th>book1</th>
      <th>chapter1</th>
      <th>verse1</th>
      <th>NODE2</th>
      <th>TYPE2</th>
      <th>TEXT2</th>
      <th>NODE3</th>
      <th>TYPE3</th>
      <th>TEXT3</th>
      <th>function3</th>
      <th>typ3</th>
      <th>NODE4</th>
      <th>TYPE4</th>
      <th>TEXT4</th>
      <th>g_cons4</th>
      <th>g_lex4</th>
      <th>g_word4</th>
      <th>gn4</th>
      <th>...</th>
      <th>TYPE5</th>
      <th>TEXT5</th>
      <th>function5</th>
      <th>typ5</th>
      <th>NODE6</th>
      <th>TYPE6</th>
      <th>TEXT6</th>
      <th>g_cons6</th>
      <th>g_lex6</th>
      <th>g_word6</th>
      <th>gn6</th>
      <th>lex6</th>
      <th>lex_utf86</th>
      <th>macula_BH_lxx_lex_16</th>
      <th>macula_BH_lxx_lex_26</th>
      <th>macula_BH_lxx_lex_36</th>
      <th>macula_BH_lxx_text6</th>
      <th>nu6</th>
      <th>ps6</th>
      <th>sp6</th>
      <th>tov_BH_text6</th>
      <th>tov_LXX_lex6</th>
      <th>tov_LXX_text6</th>
      <th>vs6</th>
      <th>vt6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>1417673</td>
      <td>verse</td>
      <td>אִ֣ישׁ אִמֹּ֤ו וְאָבִיו֙ תִּירָ֔אוּ וְאֶת־שַׁב...</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>439829</td>
      <td>clause</td>
      <td>אִמֹּ֤ו וְאָבִיו֙ תִּירָ֔אוּ</td>
      <td>688881</td>
      <td>phrase</td>
      <td>תִּירָ֔אוּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>63851</td>
      <td>word</td>
      <td>תִּירָ֔אוּ</td>
      <td>TJR&gt;W</td>
      <td>JR@&gt;</td>
      <td>T.IJR@80&gt;W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>אִמֹּ֤ו וְאָבִיו֙</td>
      <td>Objc</td>
      <td>NP</td>
      <td>63848</td>
      <td>word</td>
      <td>אִמֹּ֤ו</td>
      <td>&gt;MW</td>
      <td>&gt;IM.</td>
      <td>&gt;IM.O70W</td>
      <td>f</td>
      <td>&gt;M/</td>
      <td>אם</td>
      <td>πατήρ</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>πατέρα|αὐτοῦ</td>
      <td>sg</td>
      <td>NaN</td>
      <td>subs</td>
      <td>אמ\ו ~אבי\ו</td>
      <td>πατήρ</td>
      <td>πατέρα αὐτοῦ</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>30</td>
      <td>1417700</td>
      <td>verse</td>
      <td>אֶת־שַׁבְּתֹתַ֣י תִּשְׁמֹ֔רוּ וּמִקְדָּשִׁ֖י ת...</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>30</td>
      <td>439925</td>
      <td>clause</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>689186</td>
      <td>phrase</td>
      <td>תִּירָ֑אוּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>64278</td>
      <td>word</td>
      <td>תִּירָ֑אוּ</td>
      <td>TJR&gt;W</td>
      <td>JR@&gt;</td>
      <td>T.IJR@92&gt;W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>Objc</td>
      <td>NP</td>
      <td>64277</td>
      <td>word</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>MQDCJ</td>
      <td>MIQ:D.@C</td>
      <td>MIQ:D.@CI73J</td>
      <td>m</td>
      <td>MQDC/</td>
      <td>מקדשׁ</td>
      <td>ἅγιος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἁγίων|μου</td>
      <td>sg</td>
      <td>NaN</td>
      <td>subs</td>
      <td>מקדשׁ\יו\מ\קדשׁ\י V</td>
      <td>ἅγιος</td>
      <td>τῶν ἁγίων μου</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Leviticus</td>
      <td>26</td>
      <td>2</td>
      <td>1417915</td>
      <td>verse</td>
      <td>אֶת־שַׁבְּתֹתַ֣י תִּשְׁמֹ֔רוּ וּמִקְדָּשִׁ֖י ת...</td>
      <td>Leviticus</td>
      <td>26</td>
      <td>2</td>
      <td>440669</td>
      <td>clause</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>691449</td>
      <td>phrase</td>
      <td>תִּירָ֑אוּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>68178</td>
      <td>word</td>
      <td>תִּירָ֑אוּ</td>
      <td>TJR&gt;W</td>
      <td>JR@&gt;</td>
      <td>T.IJR@92&gt;W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>Objc</td>
      <td>NP</td>
      <td>68177</td>
      <td>word</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>MQDCJ</td>
      <td>MIQ:D.@C</td>
      <td>MIQ:D.@CI73J</td>
      <td>m</td>
      <td>MQDC/</td>
      <td>מקדשׁ</td>
      <td>ἅγιος</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ἁγίων|μου</td>
      <td>sg</td>
      <td>NaN</td>
      <td>subs</td>
      <td>מקדשׁ\יו\מ\קדשׁי V prp+</td>
      <td>ἅγιος</td>
      <td>τῶν ἁγίων μου</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Deuteronomy</td>
      <td>25</td>
      <td>18</td>
      <td>1419955</td>
      <td>verse</td>
      <td>אֲשֶׁ֨ר קָֽרְךָ֜ בַּדֶּ֗רֶךְ וַיְזַנֵּ֤ב בְּךָ...</td>
      <td>Deuteronomium</td>
      <td>25</td>
      <td>18</td>
      <td>447594</td>
      <td>clause</td>
      <td>וְלֹ֥א יָרֵ֖א אֱלֹהִֽים׃</td>
      <td>712454</td>
      <td>phrase</td>
      <td>יָרֵ֖א</td>
      <td>Pred</td>
      <td>VP</td>
      <td>107125</td>
      <td>word</td>
      <td>יָרֵ֖א</td>
      <td>JR&gt;</td>
      <td>J@R;&gt;</td>
      <td>J@R;73&gt;</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>אֱלֹהִֽים׃</td>
      <td>Objc</td>
      <td>NP</td>
      <td>107126</td>
      <td>word</td>
      <td>אֱלֹהִֽים׃</td>
      <td>&gt;LHJM</td>
      <td>&gt;:ELOH</td>
      <td>&gt;:ELOHI75JM</td>
      <td>m</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>θεόν</td>
      <td>pl</td>
      <td>NaN</td>
      <td>subs</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>τὸν θεόν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>2_Kings</td>
      <td>17</td>
      <td>7</td>
      <td>1424297</td>
      <td>verse</td>
      <td>וַיְהִ֗י כִּֽי־חָטְא֤וּ בְנֵֽי־יִשְׂרָאֵל֙ לַי...</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>7</td>
      <td>467059</td>
      <td>clause</td>
      <td>וַיִּֽירְא֖וּ אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>770760</td>
      <td>phrase</td>
      <td>יִּֽירְא֖וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>205722</td>
      <td>word</td>
      <td>יִּֽירְא֖וּ</td>
      <td>JJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>J.I45JR:&gt;73W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>Objc</td>
      <td>NP</td>
      <td>205723</td>
      <td>word</td>
      <td>אֱלֹהִ֥ים</td>
      <td>&gt;LHJM</td>
      <td>&gt;:ELOH</td>
      <td>&gt;:ELOHI71JM</td>
      <td>m</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>θεοὺς</td>
      <td>pl</td>
      <td>NaN</td>
      <td>subs</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>θεοὺς</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>2_Kings</td>
      <td>17</td>
      <td>35</td>
      <td>1424325</td>
      <td>verse</td>
      <td>וַיִּכְרֹ֨ת יְהוָ֤ה אִתָּם֙ בְּרִ֔ית וַיְצַוֵּ...</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>35</td>
      <td>467176</td>
      <td>clause</td>
      <td>לֹ֥א תִֽירְא֖וּ אֱלֹהִ֣ים אֲחֵרִ֑ים</td>
      <td>771150</td>
      <td>phrase</td>
      <td>תִֽירְא֖וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>206390</td>
      <td>word</td>
      <td>תִֽירְא֖וּ</td>
      <td>TJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>TI45JR:&gt;73W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>אֱלֹהִ֣ים אֲחֵרִ֑ים</td>
      <td>Objc</td>
      <td>NP</td>
      <td>206391</td>
      <td>word</td>
      <td>אֱלֹהִ֣ים</td>
      <td>&gt;LHJM</td>
      <td>&gt;:ELOH</td>
      <td>&gt;:ELOHI74JM</td>
      <td>m</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>θεοὺς</td>
      <td>pl</td>
      <td>NaN</td>
      <td>subs</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>θεοὺς</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>2_Kings</td>
      <td>17</td>
      <td>37</td>
      <td>1424327</td>
      <td>verse</td>
      <td>וְאֶת־הַחֻקִּ֨ים וְאֶת־הַמִּשְׁפָּטִ֜ים וְהַתּ...</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>37</td>
      <td>467188</td>
      <td>clause</td>
      <td>וְלֹ֥א תִֽירְא֖וּ אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>771188</td>
      <td>phrase</td>
      <td>תִֽירְא֖וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>206454</td>
      <td>word</td>
      <td>תִֽירְא֖וּ</td>
      <td>TJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>TI45JR:&gt;73W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>Objc</td>
      <td>NP</td>
      <td>206455</td>
      <td>word</td>
      <td>אֱלֹהִ֥ים</td>
      <td>&gt;LHJM</td>
      <td>&gt;:ELOH</td>
      <td>&gt;:ELOHI71JM</td>
      <td>m</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>θεοὺς</td>
      <td>pl</td>
      <td>NaN</td>
      <td>subs</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>θεοὺς</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>2_Kings</td>
      <td>17</td>
      <td>38</td>
      <td>1424328</td>
      <td>verse</td>
      <td>וְהַבְּרִ֛ית אֲשֶׁר־כָּרַ֥תִּי אִתְּכֶ֖ם לֹ֣א ...</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>38</td>
      <td>467191</td>
      <td>clause</td>
      <td>וְלֹ֥א תִֽירְא֖וּ אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>771199</td>
      <td>phrase</td>
      <td>תִֽירְא֖וּ</td>
      <td>Pred</td>
      <td>VP</td>
      <td>206467</td>
      <td>word</td>
      <td>תִֽירְא֖וּ</td>
      <td>TJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>TI45JR:&gt;73W.</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>Objc</td>
      <td>NP</td>
      <td>206468</td>
      <td>word</td>
      <td>אֱלֹהִ֥ים</td>
      <td>&gt;LHJM</td>
      <td>&gt;:ELOH</td>
      <td>&gt;:ELOHI71JM</td>
      <td>m</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>θεοὺς</td>
      <td>pl</td>
      <td>NaN</td>
      <td>subs</td>
      <td>אלהים</td>
      <td>θεός</td>
      <td>θεοὺς</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Isaiah</td>
      <td>51</td>
      <td>7</td>
      <td>1425585</td>
      <td>verse</td>
      <td>שִׁמְע֤וּ אֵלַי֙ יֹ֣דְעֵי צֶ֔דֶק עַ֖ם תֹּורָתִ...</td>
      <td>Jesaia</td>
      <td>51</td>
      <td>7</td>
      <td>472552</td>
      <td>clause</td>
      <td>אַל־תִּֽירְאוּ֙ חֶרְפַּ֣ת אֱנֹ֔ושׁ</td>
      <td>785854</td>
      <td>phrase</td>
      <td>תִּֽירְאוּ֙</td>
      <td>Pred</td>
      <td>VP</td>
      <td>229985</td>
      <td>word</td>
      <td>תִּֽירְאוּ֙</td>
      <td>TJR&gt;W</td>
      <td>JR:&gt;</td>
      <td>T.I45JR:&gt;W.03</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>חֶרְפַּ֣ת אֱנֹ֔ושׁ</td>
      <td>Objc</td>
      <td>NP</td>
      <td>229986</td>
      <td>word</td>
      <td>חֶרְפַּ֣ת</td>
      <td>XRPT</td>
      <td>XER:P.</td>
      <td>XER:P.A74T</td>
      <td>f</td>
      <td>XRPH/</td>
      <td>חרפה</td>
      <td>ὀνειδισμός</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ὀνειδισμὸν</td>
      <td>sg</td>
      <td>NaN</td>
      <td>subs</td>
      <td>חרפת</td>
      <td>ὀνειδισμός</td>
      <td>ὀνειδισμὸν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Ezekiel</td>
      <td>11</td>
      <td>8</td>
      <td>1427413</td>
      <td>verse</td>
      <td>חֶ֖רֶב יְרֵאתֶ֑ם וְחֶ֨רֶב֙ אָבִ֣יא עֲלֵיכֶ֔ם נ...</td>
      <td>Ezechiel</td>
      <td>11</td>
      <td>8</td>
      <td>480961</td>
      <td>clause</td>
      <td>חֶ֖רֶב יְרֵאתֶ֑ם</td>
      <td>809431</td>
      <td>phrase</td>
      <td>יְרֵאתֶ֑ם</td>
      <td>Pred</td>
      <td>VP</td>
      <td>268989</td>
      <td>word</td>
      <td>יְרֵאתֶ֑ם</td>
      <td>JR&gt;TM</td>
      <td>J:R;&gt;</td>
      <td>J:R;&gt;TE92M</td>
      <td>m</td>
      <td>...</td>
      <td>phrase</td>
      <td>חֶ֖רֶב</td>
      <td>Objc</td>
      <td>NP</td>
      <td>268988</td>
      <td>word</td>
      <td>חֶ֖רֶב</td>
      <td>XRB</td>
      <td>XEREB</td>
      <td>XE73REB</td>
      <td>f</td>
      <td>XRB/</td>
      <td>חרב</td>
      <td>ῥομφαία</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>ῥομφαίαν</td>
      <td>sg</td>
      <td>NaN</td>
      <td>subs</td>
      <td>חרב</td>
      <td>ῥομφαία</td>
      <td>ῥομφαίαν</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 65 columns</p>
</div>




```python
# Rename columns
FearValenceDO1 = FearValenceDO1.rename(columns={

"R":"order",
"book1":"book",
"chapter1":"chapter",
"verse1":"verse",
"NODE2":"clause #",
"TEXT2":"verse_text",



"NODE3":"phrase # Pred",
"function3":"function Pred",
"typ3":"phrase type Pred",
"TEXT3":"phrase text Pred",

"NODE4":"word H # Pred",
"TEXT4":"word H text Pred",
"sp4":"word H Pred part of speech",
"nu4":"word H Pred number",
"ps4":"word H Pred person",
"gn4":"word H Pred gender",
"vs4":"word H Pred stem",
"vt4":"word H Pred tense",
"lex4":"word H lex Pred",
"lex_utf84":"word H lexUTF8 Pred",
"tov_LXX_text4":"word G text Pred",
"tov_LXX_lex4":"word G lex Pred",



"NODE5":"phrase # Cmpl",
"function5":"function Cmpl",
"typ5":"phrase type Cmpl",
"TEXT5":"phrase text Cmpl",

"NODE6":"word H # Cmpl-element-2",
"TEXT6":"word H text Cmpl-element-2",
"sp6":"word H Cmpl-element-2 part of speech",
"nu6":"word H Cmpl-element-2 number",
"ps6":"word H Cmpl-element-2 person",
"gn6":"word H Cmpl-element-2 gender",
"vs6":"word H Cmpl-element-2 stem",
"vt6":"word H Cmpl-element-2 tense",
"lex6":"word H lex Cmpl-element-2",
"lex_utf86":"word H lexUTF8 Cmpl-element-2",
"tov_LXX_text6":"word G text Cmpl-element-2",
"tov_LXX_lex6":"word G lex Cmpl-element-2",


})

# Reduce DataFrame to the selected columns in the desired order
FearValenceDO1 = FearValenceDO1[
[

"order",
"book",
"chapter",
"verse",
"clause #",
"verse_text",



"phrase # Pred",
"function Pred",
"phrase type Pred",
"phrase text Pred",

"word H # Pred",
"word H text Pred",
"word H Pred part of speech",
"word H Pred number",
"word H Pred person",
"word H Pred gender",
"word H Pred stem",
"word H Pred tense",
"word H lex Pred",
"word H lexUTF8 Pred",
"word G text Pred",
"word G lex Pred",



"phrase # Cmpl",
"function Cmpl",
"phrase type Cmpl",
"phrase text Cmpl",

"word H # Cmpl-element-2",
"word H text Cmpl-element-2",
"word H Cmpl-element-2 part of speech",
"word H Cmpl-element-2 number",
"word H Cmpl-element-2 person",
"word H Cmpl-element-2 gender",
"word H Cmpl-element-2 stem",
"word H Cmpl-element-2 tense",
"word H lex Cmpl-element-2",
"word H lexUTF8 Cmpl-element-2",
"word G text Cmpl-element-2",
"word G lex Cmpl-element-2",



]
]
FearValenceDO1.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
      <th>phrase # Cmpl</th>
      <th>function Cmpl</th>
      <th>phrase type Cmpl</th>
      <th>phrase text Cmpl</th>
      <th>word H # Cmpl-element-2</th>
      <th>word H text Cmpl-element-2</th>
      <th>word H Cmpl-element-2 part of speech</th>
      <th>word H Cmpl-element-2 number</th>
      <th>word H Cmpl-element-2 person</th>
      <th>word H Cmpl-element-2 gender</th>
      <th>word H Cmpl-element-2 stem</th>
      <th>word H Cmpl-element-2 tense</th>
      <th>word H lex Cmpl-element-2</th>
      <th>word H lexUTF8 Cmpl-element-2</th>
      <th>word G text Cmpl-element-2</th>
      <th>word G lex Cmpl-element-2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>439829</td>
      <td>אִמֹּ֤ו וְאָבִיו֙ תִּירָ֔אוּ</td>
      <td>688881</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֔אוּ</td>
      <td>63851</td>
      <td>תִּירָ֔אוּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβείσθω</td>
      <td>φοβέω</td>
      <td>688880</td>
      <td>Objc</td>
      <td>NP</td>
      <td>אִמֹּ֤ו וְאָבִיו֙</td>
      <td>63848</td>
      <td>אִמֹּ֤ו</td>
      <td>subs</td>
      <td>sg</td>
      <td>NaN</td>
      <td>f</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;M/</td>
      <td>אם</td>
      <td>πατέρα αὐτοῦ</td>
      <td>πατήρ</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>30</td>
      <td>439925</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>689186</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֑אוּ</td>
      <td>64278</td>
      <td>תִּירָ֑אוּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβηθήσεσθε</td>
      <td>φοβέω</td>
      <td>689185</td>
      <td>Objc</td>
      <td>NP</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>64277</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>subs</td>
      <td>sg</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MQDC/</td>
      <td>מקדשׁ</td>
      <td>τῶν ἁγίων μου</td>
      <td>ἅγιος</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Leviticus</td>
      <td>26</td>
      <td>2</td>
      <td>440669</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>691449</td>
      <td>Pred</td>
      <td>VP</td>
      <td>תִּירָ֑אוּ</td>
      <td>68178</td>
      <td>תִּירָ֑אוּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβηθήσεσθε</td>
      <td>φοβέω</td>
      <td>691448</td>
      <td>Objc</td>
      <td>NP</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>68177</td>
      <td>מִקְדָּשִׁ֖י</td>
      <td>subs</td>
      <td>sg</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>MQDC/</td>
      <td>מקדשׁ</td>
      <td>τῶν ἁγίων μου</td>
      <td>ἅγιος</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Deuteronomium</td>
      <td>25</td>
      <td>18</td>
      <td>447594</td>
      <td>וְלֹ֥א יָרֵ֖א אֱלֹהִֽים׃</td>
      <td>712454</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יָרֵ֖א</td>
      <td>107125</td>
      <td>יָרֵ֖א</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>perf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθη</td>
      <td>φοβέω</td>
      <td>712455</td>
      <td>Objc</td>
      <td>NP</td>
      <td>אֱלֹהִֽים׃</td>
      <td>107126</td>
      <td>אֱלֹהִֽים׃</td>
      <td>subs</td>
      <td>pl</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>τὸν θεόν</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>7</td>
      <td>467059</td>
      <td>וַיִּֽירְא֖וּ אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>770760</td>
      <td>Pred</td>
      <td>VP</td>
      <td>יִּֽירְא֖וּ</td>
      <td>205722</td>
      <td>יִּֽירְא֖וּ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>wayq</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>ἐφοβήθησαν</td>
      <td>φοβέω</td>
      <td>770761</td>
      <td>Objc</td>
      <td>NP</td>
      <td>אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>205723</td>
      <td>אֱלֹהִ֥ים</td>
      <td>subs</td>
      <td>pl</td>
      <td>NaN</td>
      <td>m</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>&gt;LHJM/</td>
      <td>אלהים</td>
      <td>θεοὺς</td>
      <td>θεός</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValenceDO1.to_excel("/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValResearchData_03_DOwoET.xlsx", index=False)
```


```python
FearValenceDO2 = '''
verse verse book chapter
  clause
    p1:phrase function=PreO typ
      word lex=JR>[ vs=qal vt=perf|impf|wayq|impv lex_utf8 gn nu ps sp g_cons g_lex g_word macula_BH_lxx_lex_1 macula_BH_lxx_lex_2 macula_BH_lxx_lex_3 macula_BH_lxx_text tov_BH_text tov_LXX_text tov_LXX_lex



'''
FearValenceDO2  = BHS.search(FearValenceDO2)
BHS.table(FearValenceDO2, start=1, end=3, condensed=False, extraFeatures='qere_utf8', fmt='text-orig-full-ketiv', colorMap={3:'yellow', 4:'magenta'})
```

      0.44s 12 results



<table><tr class="tf rtl"><th class="tf rtl">n</th><th class="tf rtl">p</th><th class="tf rtl">verse</th><th class="tf">clause</th><th class="tf">phrase</th><th class="tf">word</th></tr>
<tr class="tf rtl"><td class="tf rtl">1</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Numeri&amp;chapter=14&amp;verse=9&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Numbers 14:9">Numbers 14:9</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">אַל־</span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָאֻֽם׃ </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָאֻֽם׃ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּירָאֻֽם׃ </span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">2</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Deuteronomium&amp;chapter=3&amp;verse=22&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="Deuteronomy 3:22">Deuteronomy 3:22</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">לֹ֖א </span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירָא֑וּם </span></span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירָא֑וּם </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">תִּֽירָא֑וּם </span></span></span></td></tr>
<tr class="tf rtl"><td class="tf rtl">3</td><td class="tf rtl"><span class="tfsechead rtl"><span class="ltr"><a target="_blank" href="https://shebanq.ancient-data.org/hebrew/text?book=Reges_I&amp;chapter=8&amp;verse=40&amp;version=2021&amp;mr=m&amp;qw=q&amp;tp=txt_p&amp;tr=hb&amp;wget=v&amp;qget=v&amp;nget=vt" title="Show this on SHEBANQ" sec="1_Kings 8:40">1_Kings 8:40</a></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx" ><span class="txtu hbo">לְמַ֨עַן֙ </span><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יִֽרָא֔וּךָ </span></span></span><span class="txtu hbo">כָּל־</span><span class="txtu hbo">הַ֨</span><span class="txtu hbo">יָּמִ֔ים </span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hlbx"  style="border-color: yellow;" ><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יִֽרָא֔וּךָ </span></span></span></span></td><td class="tf rtl"><span class="rtl ubd pl-mode"><span class="plain rtl  hl"  style="background-color: magenta;" ><span class="txtu hbo">יִֽרָא֔וּךָ </span></span></span></td></tr></table>



```python
BHS.export(FearValenceDO2, toDir='/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/', toFile='FearValDataExport_05_PreO.tsv')
```


```python
FearValencePreO=pd.read_csv('/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValDataExport_05_PreO.tsv',delimiter='\t',encoding='utf-16')
pd.set_option('display.max_columns', 50)
FearValencePreO.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>NODE1</th>
      <th>TYPE1</th>
      <th>TEXT1</th>
      <th>book1</th>
      <th>chapter1</th>
      <th>verse1</th>
      <th>NODE2</th>
      <th>TYPE2</th>
      <th>TEXT2</th>
      <th>NODE3</th>
      <th>TYPE3</th>
      <th>TEXT3</th>
      <th>function3</th>
      <th>typ3</th>
      <th>NODE4</th>
      <th>TYPE4</th>
      <th>TEXT4</th>
      <th>g_cons4</th>
      <th>g_lex4</th>
      <th>g_word4</th>
      <th>gn4</th>
      <th>lex4</th>
      <th>lex_utf84</th>
      <th>macula_BH_lxx_lex_14</th>
      <th>macula_BH_lxx_lex_24</th>
      <th>macula_BH_lxx_lex_34</th>
      <th>macula_BH_lxx_text4</th>
      <th>nu4</th>
      <th>ps4</th>
      <th>sp4</th>
      <th>tov_BH_text4</th>
      <th>tov_LXX_lex4</th>
      <th>tov_LXX_text4</th>
      <th>vs4</th>
      <th>vt4</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Numbers</td>
      <td>14</td>
      <td>9</td>
      <td>1418506</td>
      <td>verse</td>
      <td>אַ֣ךְ בַּֽיהוָה֮ אַל־תִּמְרֹדוּ֒ וְאַתֶּ֗ם אַל...</td>
      <td>Numeri</td>
      <td>14</td>
      <td>9</td>
      <td>442392</td>
      <td>clause</td>
      <td>אַל־תִּירָאֻֽם׃</td>
      <td>696617</td>
      <td>phrase</td>
      <td>תִּירָאֻֽם׃</td>
      <td>PreO</td>
      <td>VP</td>
      <td>78597</td>
      <td>word</td>
      <td>תִּירָאֻֽם׃</td>
      <td>TJR&gt;M</td>
      <td>JR@&gt;</td>
      <td>T.IJR@&gt;U75M</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>αὐτός</td>
      <td>NaN</td>
      <td>φοβηθῆτε|αὐτούς</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>תירא\ם</td>
      <td>φοβέω</td>
      <td>φοβηθῆτε αὐτούς</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Deuteronomy</td>
      <td>3</td>
      <td>22</td>
      <td>1419387</td>
      <td>verse</td>
      <td>לֹ֖א תִּֽירָא֑וּם כִּ֚י יְהוָ֣ה אֱלֹֽהֵיכֶ֔ם ה...</td>
      <td>Deuteronomium</td>
      <td>3</td>
      <td>22</td>
      <td>445283</td>
      <td>clause</td>
      <td>לֹ֖א תִּֽירָא֑וּם</td>
      <td>705411</td>
      <td>phrase</td>
      <td>תִּֽירָא֑וּם</td>
      <td>PreO</td>
      <td>VP</td>
      <td>95000</td>
      <td>word</td>
      <td>תִּֽירָא֑וּם</td>
      <td>TJR&gt;WM</td>
      <td>JR@&gt;</td>
      <td>T.I45JR@&gt;92W.M</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>φοβηθήσεσθε</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>תיראו\ם</td>
      <td>φοβέω</td>
      <td>φοβηθήσεσθε</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1_Kings</td>
      <td>8</td>
      <td>40</td>
      <td>1423331</td>
      <td>verse</td>
      <td>לְמַ֨עַן֙ יִֽרָא֔וּךָ כָּל־הַ֨יָּמִ֔ים אֲשֶׁר־...</td>
      <td>Reges_I</td>
      <td>8</td>
      <td>40</td>
      <td>462381</td>
      <td>clause</td>
      <td>לְמַ֨עַן֙ יִֽרָא֔וּךָ כָּל־הַ֨יָּמִ֔ים</td>
      <td>757075</td>
      <td>phrase</td>
      <td>יִֽרָא֔וּךָ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>183004</td>
      <td>word</td>
      <td>יִֽרָא֔וּךָ</td>
      <td>JR&gt;WK</td>
      <td>R@&gt;</td>
      <td>JI45R@&gt;80W.K@</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβῶνταί|σε</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראו\ך</td>
      <td>φοβέω</td>
      <td>φοβῶνταί σε</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Isaiah</td>
      <td>25</td>
      <td>3</td>
      <td>1425026</td>
      <td>verse</td>
      <td>עַל־כֵּ֖ן יְכַבְּד֣וּךָ עַם־עָ֑ז קִרְיַ֛ת גֹּו...</td>
      <td>Jesaia</td>
      <td>25</td>
      <td>3</td>
      <td>469945</td>
      <td>clause</td>
      <td>קִרְיַ֛ת גֹּויִ֥ם עָרִיצִ֖ים יִירָאֽוּךָ׃</td>
      <td>779038</td>
      <td>phrase</td>
      <td>יִירָאֽוּךָ׃</td>
      <td>PreO</td>
      <td>VP</td>
      <td>220100</td>
      <td>word</td>
      <td>יִירָאֽוּךָ׃</td>
      <td>JJR&gt;WK</td>
      <td>JR@&gt;</td>
      <td>JIJR@&gt;75W.K@</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>εὐλογέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>εὐλογήσουσίν|σε</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>ייראו\ך</td>
      <td>εὐλογέω</td>
      <td>εὐλογήσουσίν σε</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Jeremiah</td>
      <td>10</td>
      <td>7</td>
      <td>1426112</td>
      <td>verse</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙ מֶ֣לֶךְ הַגֹּויִ֔ם כִּ֥י ...</td>
      <td>Jeremia</td>
      <td>10</td>
      <td>7</td>
      <td>475070</td>
      <td>clause</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙</td>
      <td>792664</td>
      <td>phrase</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>PreO</td>
      <td>VP</td>
      <td>240005</td>
      <td>word</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>JR&gt;K</td>
      <td>R@&gt;</td>
      <td>JI45R@&gt;:AK@03</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>’’</td>
      <td>sg</td>
      <td>p3</td>
      <td>verb</td>
      <td>ירא\ך</td>
      <td>NaN</td>
      <td>-</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>5</th>
      <td>6</td>
      <td>Malachi</td>
      <td>2</td>
      <td>5</td>
      <td>1429501</td>
      <td>verse</td>
      <td>בְּרִיתִ֣י׀ הָיְתָ֣ה אִתֹּ֗ו הַֽחַיִּים֙ וְהַ֨...</td>
      <td>Maleachi</td>
      <td>2</td>
      <td>5</td>
      <td>490060</td>
      <td>clause</td>
      <td>וַיִּֽירָאֵ֑נִי</td>
      <td>834865</td>
      <td>phrase</td>
      <td>יִּֽירָאֵ֑נִי</td>
      <td>PreO</td>
      <td>VP</td>
      <td>309875</td>
      <td>word</td>
      <td>יִּֽירָאֵ֑נִי</td>
      <td>JJR&gt;NJ</td>
      <td>JR@&gt;</td>
      <td>J.I45JR@&gt;;92NIJ</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβεῖσθαί|με</td>
      <td>sg</td>
      <td>p3</td>
      <td>verb</td>
      <td>יירא\ני</td>
      <td>φοβέω</td>
      <td>φοβεῖσθαί</td>
      <td>qal</td>
      <td>wayq</td>
    </tr>
    <tr>
      <th>6</th>
      <td>7</td>
      <td>Malachi</td>
      <td>3</td>
      <td>5</td>
      <td>1429518</td>
      <td>verse</td>
      <td>וְקָרַבְתִּ֣י אֲלֵיכֶם֮ לַמִּשְׁפָּט֒ וְהָיִ֣י...</td>
      <td>Maleachi</td>
      <td>3</td>
      <td>5</td>
      <td>490144</td>
      <td>clause</td>
      <td>וְלֹ֣א יְרֵא֔וּנִי</td>
      <td>835117</td>
      <td>phrase</td>
      <td>יְרֵא֔וּנִי</td>
      <td>PreO</td>
      <td>VP</td>
      <td>310278</td>
      <td>word</td>
      <td>יְרֵא֔וּנִי</td>
      <td>JR&gt;WNJ</td>
      <td>J:R;&gt;</td>
      <td>J:R;&gt;80W.NIJ</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβουμένους|με</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראו\ני</td>
      <td>φοβέω</td>
      <td>φοβουμένους με</td>
      <td>qal</td>
      <td>perf</td>
    </tr>
    <tr>
      <th>7</th>
      <td>8</td>
      <td>Psalms</td>
      <td>72</td>
      <td>5</td>
      <td>1430654</td>
      <td>verse</td>
      <td>יִֽירָא֥וּךָ עִם־שָׁ֑מֶשׁ וְלִפְנֵ֥י יָ֝רֵ֗חַ ...</td>
      <td>Psalmi</td>
      <td>72</td>
      <td>5</td>
      <td>493723</td>
      <td>clause</td>
      <td>יִֽירָא֥וּךָ עִם־שָׁ֑מֶשׁ וְלִפְנֵ֥י יָ֝רֵ֗חַ ...</td>
      <td>843694</td>
      <td>phrase</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>322351</td>
      <td>word</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>JJR&gt;WK</td>
      <td>JR@&gt;</td>
      <td>JI45JR@&gt;71W.K@</td>
      <td>m</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>καί</td>
      <td>συμπαραμένω</td>
      <td>NaN</td>
      <td>καὶ|συμπαραμενεῖ</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>ייראו\ךו\יאריך met</td>
      <td>καί συμπαραμένω</td>
      <td>καὶ συμπαραμενεῖ</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
    <tr>
      <th>8</th>
      <td>9</td>
      <td>Psalms</td>
      <td>119</td>
      <td>63</td>
      <td>1431623</td>
      <td>verse</td>
      <td>חָבֵ֣ר אָ֭נִי לְכָל־אֲשֶׁ֣ר יְרֵא֑וּךָ וּ֝לְשׁ...</td>
      <td>Psalmi</td>
      <td>119</td>
      <td>63</td>
      <td>496311</td>
      <td>clause</td>
      <td>אֲשֶׁ֣ר יְרֵא֑וּךָ</td>
      <td>850161</td>
      <td>phrase</td>
      <td>יְרֵא֑וּךָ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>331790</td>
      <td>word</td>
      <td>יְרֵא֑וּךָ</td>
      <td>JR&gt;WK</td>
      <td>J:R;&gt;</td>
      <td>J:R;&gt;92W.K@</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>φοβουμένων|σε</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>יראו\ך</td>
      <td>φοβέω</td>
      <td>φοβουμένων σε</td>
      <td>qal</td>
      <td>perf</td>
    </tr>
    <tr>
      <th>9</th>
      <td>10</td>
      <td>Job</td>
      <td>9</td>
      <td>35</td>
      <td>1432281</td>
      <td>verse</td>
      <td>אַֽ֭דַבְּרָה וְלֹ֣א אִירָאֶ֑נּוּ כִּ֥י לֹא־כֵ֥...</td>
      <td>Iob</td>
      <td>9</td>
      <td>35</td>
      <td>498186</td>
      <td>clause</td>
      <td>וְלֹ֣א אִירָאֶ֑נּוּ</td>
      <td>854986</td>
      <td>phrase</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>338580</td>
      <td>word</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>&gt;JR&gt;NW</td>
      <td>JR@&gt;</td>
      <td>&gt;IJR@&gt;E92N.W.</td>
      <td>unknown</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβέω</td>
      <td>φοβέω</td>
      <td>NaN</td>
      <td>φοβηθῶ</td>
      <td>sg</td>
      <td>p1</td>
      <td>verb</td>
      <td>אירא\נו</td>
      <td>φοβέω</td>
      <td>φοβηθῶ</td>
      <td>qal</td>
      <td>impf</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Rename columns
FearValencePreO = FearValencePreO.rename(columns={

"R":"order",
"book1":"book",
"chapter1":"chapter",
"verse1":"verse",
"NODE2":"clause #",
"TEXT2":"verse_text",



"NODE3":"phrase # Pred",
"function3":"function Pred",
"typ3":"phrase type Pred",
"TEXT3":"phrase text Pred",

"NODE4":"word H # Pred",
"TEXT4":"word H text Pred",
"sp4":"word H Pred part of speech",
"nu4":"word H Pred number",
"ps4":"word H Pred person",
"gn4":"word H Pred gender",
"vs4":"word H Pred stem",
"vt4":"word H Pred tense",
"lex4":"word H lex Pred",
"lex_utf84":"word H lexUTF8 Pred",
"tov_LXX_text4":"word G text Pred",
"tov_LXX_lex4":"word G lex Pred",



"NODE5":"phrase # Cmpl",
"function5":"function Cmpl",
"typ5":"phrase type Cmpl",
"TEXT5":"phrase text Cmpl",

"NODE6":"word H # Cmpl-element-1",
"TEXT6":"word H text Cmpl-element-1",
"sp6":"word H Cmpl-element-1 part of speech",
"nu6":"word H Cmpl-element-1 number",
"ps6":"word H Cmpl-element-1 person",
"gn6":"word H Cmpl-element-1 gender",
"vs6":"word H Cmpl-element-1 stem",
"vt6":"word H Cmpl-element-1 tense",
"lex6":"word H lex Cmpl-element-1",
"lex_utf86":"word H lexUTF8 Cmpl-element-1",
"tov_LXX_text6":"word G text Cmpl-element-1",
"tov_LXX_lex6":"word G lex Cmpl-element-1",

"NODE7":"word H # Cmpl-element-2",
"TEXT7":"word H text Cmpl-element-2",
"sp7":"word H Cmpl-element-2 part of speech",
"nu7":"word H Cmpl-element-2 number",
"ps7":"word H Cmpl-element-2 person",
"gn7":"word H Cmpl-element-2 gender",
"vs7":"word H Cmpl-element-2 stem",
"vt7":"word H Cmpl-element-2 tense",
"lex7":"word H lex Cmpl-element-2",
"lex_utf87":"word H lexUTF8 Cmpl-element-2",
"tov_LXX_text7":"word G text Cmpl-element-2",
"tov_LXX_lex7":"word G lex Cmpl-element-2",


})

# Reduce DataFrame to the selected columns in the desired order
FearValencePreO = FearValencePreO[
[

"order",
"book",
"chapter",
"verse",
"clause #",
"verse_text",



"phrase # Pred",
"function Pred",
"phrase type Pred",
"phrase text Pred",

"word H # Pred",
"word H text Pred",
"word H Pred part of speech",
"word H Pred number",
"word H Pred person",
"word H Pred gender",
"word H Pred stem",
"word H Pred tense",
"word H lex Pred",
"word H lexUTF8 Pred",
"word G text Pred",
"word G lex Pred",



]
]
FearValencePreO.head()


```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>phrase text Pred</th>
      <th>word H # Pred</th>
      <th>word H text Pred</th>
      <th>word H Pred part of speech</th>
      <th>word H Pred number</th>
      <th>word H Pred person</th>
      <th>word H Pred gender</th>
      <th>word H Pred stem</th>
      <th>word H Pred tense</th>
      <th>word H lex Pred</th>
      <th>word H lexUTF8 Pred</th>
      <th>word G text Pred</th>
      <th>word G lex Pred</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>Numeri</td>
      <td>14</td>
      <td>9</td>
      <td>442392</td>
      <td>אַל־תִּירָאֻֽם׃</td>
      <td>696617</td>
      <td>PreO</td>
      <td>VP</td>
      <td>תִּירָאֻֽם׃</td>
      <td>78597</td>
      <td>תִּירָאֻֽם׃</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβηθῆτε αὐτούς</td>
      <td>φοβέω</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Deuteronomium</td>
      <td>3</td>
      <td>22</td>
      <td>445283</td>
      <td>לֹ֖א תִּֽירָא֑וּם</td>
      <td>705411</td>
      <td>PreO</td>
      <td>VP</td>
      <td>תִּֽירָא֑וּם</td>
      <td>95000</td>
      <td>תִּֽירָא֑וּם</td>
      <td>verb</td>
      <td>pl</td>
      <td>p2</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβηθήσεσθε</td>
      <td>φοβέω</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Reges_I</td>
      <td>8</td>
      <td>40</td>
      <td>462381</td>
      <td>לְמַ֨עַן֙ יִֽרָא֔וּךָ כָּל־הַ֨יָּמִ֔ים</td>
      <td>757075</td>
      <td>PreO</td>
      <td>VP</td>
      <td>יִֽרָא֔וּךָ</td>
      <td>183004</td>
      <td>יִֽרָא֔וּךָ</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>φοβῶνταί σε</td>
      <td>φοβέω</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Jesaia</td>
      <td>25</td>
      <td>3</td>
      <td>469945</td>
      <td>קִרְיַ֛ת גֹּויִ֥ם עָרִיצִ֖ים יִירָאֽוּךָ׃</td>
      <td>779038</td>
      <td>PreO</td>
      <td>VP</td>
      <td>יִירָאֽוּךָ׃</td>
      <td>220100</td>
      <td>יִירָאֽוּךָ׃</td>
      <td>verb</td>
      <td>pl</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>εὐλογήσουσίν σε</td>
      <td>εὐλογέω</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Jeremia</td>
      <td>10</td>
      <td>7</td>
      <td>475070</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙</td>
      <td>792664</td>
      <td>PreO</td>
      <td>VP</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>240005</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>verb</td>
      <td>sg</td>
      <td>p3</td>
      <td>m</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>-</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
FearValencePreO.to_excel("/Users/oliverglanz/Library/CloudStorage/OneDrive-AndrewsUniversity/4001_publication-presentation/0_pub-article_2026_HIPHIL_Fear/data_export/FearValResearchData_04_DOasPreO.xlsx", index=False)
```


```python
lex7_counts = FearValencePreO["word G lex Pred"].value_counts()
lex7_counts
```




    word G lex Pred
    φοβέω              9
    εὐλογέω            1
    καί συμπαραμένω    1
    Name: count, dtype: int64




```python
import matplotlib as mpl
import matplotlib.pyplot as plt

# Styling
mpl.rcParams["font.family"] = "Courier New"
mpl.rcParams["font.weight"] = "bold"
mpl.rcParams["axes.labelweight"] = "bold"
mpl.rcParams["axes.titleweight"] = "bold"
mpl.rcParams["pdf.fonttype"] = 42
mpl.rcParams["ps.fonttype"] = 42

plt.figure(figsize=(7, 4))

bars = plt.bar(lex7_counts.index.astype(str), lex7_counts.values)

plt.xlabel("word G lex Pred")
plt.ylabel("Count")
plt.title("Distribution of lex7 values")

# Add count labels
for bar in bars:
    h = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        h,
        str(int(h)),
        ha="center",
        va="bottom"
    )

plt.xticks(rotation=30, ha="right")

# Save outputs
plt.savefig("fig07_lex7_distribution.pdf", bbox_inches="tight")
plt.savefig("fig07_lex7_distribution.png", dpi=600, bbox_inches="tight")

plt.show()

```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_64_0.png)
    



```python

```

# MT vs GT valence

## Combining Valence Data Sources


```python
from pathlib import Path

import pandas as pd
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

data_export_dir = Path.cwd() / "data_export"
output_file = data_export_dir / "FearValResearchData_combined.xlsx"
wo_art_output_file = data_export_dir / "FearValResearchData_combined_woArt.xlsx"

# This precedence reproduces the current manually combined research file.
# When a clause occurs in more than one source, the first source listed wins.
source_patterns = [
    "FearValResearchData_03_*.xlsx",
    "FearValResearchData_02_*.xlsx",
    "FearValResearchData_04_*.xlsx",
    "FearValResearchData_01_*.xlsx",
]

source_files = []
for pattern in source_patterns:
    matches = sorted(data_export_dir.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one source matching {pattern!r}; found {len(matches)}"
        )
    source_files.append(matches[0])

source_frames = {
    source_file: pd.read_excel(source_file)
    for source_file in source_files
}

# Use the complete header and column order from the widest source workbook.
header_source = max(source_files, key=lambda path: len(source_frames[path].columns))
master_columns = source_frames[header_source].columns.tolist()
if "file name" not in master_columns:
    master_columns.insert(0, "file name")

# Stop rather than silently discard a new field introduced by a narrower file.
unexpected_columns = {
    source_file.name: sorted(set(frame.columns) - set(master_columns))
    for source_file, frame in source_frames.items()
    if set(frame.columns) - set(master_columns)
}
if unexpected_columns:
    raise ValueError(f"Headers absent from the widest source: {unexpected_columns}")

element1_id_column = "word H # Cmpl-element-1"
element2_id_column = "word H # Cmpl-element-2"
element2_start = master_columns.index(element2_id_column)
element2_end = master_columns.index("word G lex Cmpl-element-2") + 1
element2_columns = master_columns[element2_start:element2_end]

# Determine how many complement-word blocks are required by the current data.
max_complement_element = 2
for source_file, source_frame in source_frames.items():
    if not {element1_id_column, element2_id_column}.issubset(source_frame.columns):
        continue
    for clause_number, clause_rows in source_frame.groupby("clause #", sort=False):
        if len(clause_rows) < 2:
            continue
        anchors = clause_rows[["phrase # Cmpl", element1_id_column]].drop_duplicates()
        if len(anchors) != 1:
            raise ValueError(
                f"Clause {clause_number} in {source_file.name} has competing complement anchors"
            )
        element1_number = pd.to_numeric(
            clause_rows[element1_id_column].iloc[0], errors="coerce"
        )
        candidate_numbers = pd.to_numeric(
            clause_rows[element2_id_column], errors="coerce"
        )
        higher_nodes = sorted(
            candidate_numbers[candidate_numbers > element1_number].dropna().astype(int).unique()
        )
        expected_nodes = list(
            range(int(element1_number) + 1, int(element1_number) + 1 + len(higher_nodes))
        )
        if higher_nodes != expected_nodes:
            raise ValueError(
                f"Nonconsecutive complement nodes in clause {clause_number}: {higher_nodes}"
            )
        max_complement_element = max(max_complement_element, 1 + len(higher_nodes))

element_block_columns = {2: element2_columns}
for element_number in range(3, max_complement_element + 1):
    element_block_columns[element_number] = [
        column.replace("Cmpl-element-2", f"Cmpl-element-{element_number}")
        for column in element2_columns
    ]

additional_element_columns = [
    column
    for element_number in range(3, max_complement_element + 1)
    for column in element_block_columns[element_number]
]
master_columns = (
    master_columns[:element2_end]
    + additional_element_columns
    + master_columns[element2_end:]
)

def collapse_source_clauses(frame):
    """Consolidate repeated clause rows into ordered complement-word blocks."""
    if frame["clause #"].isna().any():
        raise ValueError("Blank values found in 'clause #'; consolidation is ambiguous")

    retained_rows = []
    higher_nodes_placed = 0
    cleared_single_word_element2 = 0
    highest_element_populated = 1

    for clause_number, clause_rows in frame.groupby("clause #", sort=False):
        retained = clause_rows.iloc[0].copy()
        if len(clause_rows) > 1 and pd.notna(retained[element1_id_column]):
            element1_number = pd.to_numeric(
                retained[element1_id_column], errors="coerce"
            )
            candidate_numbers = pd.to_numeric(
                clause_rows[element2_id_column], errors="coerce"
            )
            higher_candidates = (
                clause_rows.loc[candidate_numbers > element1_number]
                .assign(_element2_number=candidate_numbers[candidate_numbers > element1_number])
                .sort_values("_element2_number")
                .drop_duplicates(subset="_element2_number", keep="first")
            )
            higher_nodes = higher_candidates["_element2_number"].astype(int).tolist()
            expected_nodes = list(
                range(int(element1_number) + 1, int(element1_number) + 1 + len(higher_nodes))
            )
            if higher_nodes != expected_nodes:
                raise ValueError(
                    f"Nonconsecutive complement nodes in clause {clause_number}: {higher_nodes}"
                )

            # Clear all generated blocks, then fill them from the ordered unique nodes.
            for target_columns in element_block_columns.values():
                retained[target_columns] = pd.NA
            for element_number, (_, candidate) in enumerate(
                higher_candidates.iterrows(), start=2
            ):
                retained[element_block_columns[element_number]] = (
                    candidate[element2_columns].to_numpy()
                )

            higher_nodes_placed += len(higher_candidates)
            if higher_candidates.empty:
                cleared_single_word_element2 += 1
            else:
                highest_element_populated = max(
                    highest_element_populated, 1 + len(higher_candidates)
                )

        retained_rows.append(retained)

    return (
        pd.DataFrame(retained_rows).reset_index(drop=True),
        higher_nodes_placed,
        cleared_single_word_element2,
        highest_element_populated,
    )

prepared_frames = []
repair_summary = {}
source_row_count = 0
for source_file in source_files:
    frame = source_frames[source_file].dropna(how="all").copy()
    source_row_count += len(frame)
    frame["file name"] = source_file.stem
    frame = frame.reindex(columns=master_columns)
    frame, placed_count, cleared_count, highest_element = collapse_source_clauses(frame)
    prepared_frames.append(frame)
    repair_summary[source_file.stem] = {
        "higher complement nodes placed": placed_count,
        "cleared single-word element-2": cleared_count,
        "highest complement element": highest_element,
    }

all_records = pd.concat(prepared_frames, ignore_index=True)
if "clause #" not in all_records.columns:
    raise KeyError("The widest source workbook does not contain 'clause #'")
if all_records["clause #"].isna().any():
    raise ValueError("Blank values found in 'clause #'; deduplication would be ambiguous")

rows_after_source_consolidation = len(all_records)
FearValResearchData_combined = (
    all_records.drop_duplicates(subset="clause #", keep="first")
    .reset_index(drop=True)
)

def remove_articles_and_shift_elements(frame):
    """Remove article word blocks and shift later complement elements left."""
    result = frame.copy()
    all_element_columns = [
        column for column in result.columns if "Cmpl-element-" in column
    ]
    # These blocks mix text, numeric identifiers, and blanks after shifting.
    result[all_element_columns] = result[all_element_columns].astype(object)
    article_elements_removed = 0
    affected_clauses = 0

    for row_index, row in result.iterrows():
        retained_blocks = []
        row_articles_removed = 0
        for source_element in range(1, max_complement_element + 1):
            source_columns = [
                column.replace("Cmpl-element-2", f"Cmpl-element-{source_element}")
                for column in element2_columns
            ]
            part_of_speech_column = (
                f"word H Cmpl-element-{source_element} part of speech"
            )
            part_of_speech = row.get(part_of_speech_column)
            if pd.notna(part_of_speech) and str(part_of_speech).strip().lower() == "art":
                article_elements_removed += 1
                row_articles_removed += 1
                continue
            if row[source_columns].notna().any():
                retained_blocks.append(row[source_columns].to_numpy())

        if row_articles_removed:
            affected_clauses += 1

            # Only affected clauses are compacted. Rows without an article retain
            # their original element positions exactly.
            for target_element in range(1, max_complement_element + 1):
                target_columns = [
                    column.replace("Cmpl-element-2", f"Cmpl-element-{target_element}")
                    for column in element2_columns
                ]
                result.loc[row_index, target_columns] = pd.NA
            for target_element, block_values in enumerate(retained_blocks, start=1):
                target_columns = [
                    column.replace("Cmpl-element-2", f"Cmpl-element-{target_element}")
                    for column in element2_columns
                ]
                result.loc[row_index, target_columns] = block_values

    return result, article_elements_removed, affected_clauses

FearValResearchData_combined_woArt, articles_removed, clauses_with_articles = (
    remove_articles_and_shift_elements(FearValResearchData_combined)
)

def write_combined_workbook(frame, destination):
    """Write a combined-data variant with the shared research-data formatting."""
    with pd.ExcelWriter(destination, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Sheet1", index=False)
        worksheet = writer.book["Sheet1"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        # Give every complement-element block its own header/body color pair.
        element_color_pairs = [
            ("9DC3E6", "DDEBF7"),  # blue
            ("F4B183", "FCE4D6"),  # orange
            ("A9D18E", "E2F0D9"),  # green
            ("B4A7D6", "E4DFEC"),  # purple
            ("F4D35E", "FFF2CC"),  # yellow
            ("80CBC4", "DDEBF7"),  # teal
            ("E6B8C7", "F4CCCC"),  # rose
            ("BFBFBF", "E7E6E6"),  # gray
        ]
        for element_number in range(1, max_complement_element + 1):
            header_color, body_color = element_color_pairs[
                (element_number - 1) % len(element_color_pairs)
            ]
            header_fill = PatternFill("solid", fgColor=header_color)
            body_fill = PatternFill("solid", fgColor=body_color)
            element_marker = f"Cmpl-element-{element_number}"
            element_column_numbers = [
                column_number
                for column_number, column_name in enumerate(frame.columns, start=1)
                if element_marker in column_name
            ]
            for column_number in element_column_numbers:
                worksheet.cell(1, column_number).fill = header_fill
                for row_number in range(2, worksheet.max_row + 1):
                    worksheet.cell(row_number, column_number).fill = body_fill

        # Size columns from data values only; header length is intentionally ignored.
        for column_number, column_name in enumerate(frame.columns, start=1):
            sample = frame[column_name].dropna().astype(str)
            content_width = sample.str.len().max() if not sample.empty else 0
            worksheet.column_dimensions[get_column_letter(column_number)].width = min(
                40, max(3, content_width + 2)
            )

write_combined_workbook(FearValResearchData_combined, output_file)
write_combined_workbook(FearValResearchData_combined_woArt, wo_art_output_file)

print(f"Created {output_file.name}")
print(f"Created {wo_art_output_file.name}")
print(f"Article elements removed from second version: {articles_removed:,} across {clauses_with_articles:,} clauses")
print(f"Header source: {header_source.name} ({len(master_columns)} columns)")
print(f"Input rows: {source_row_count:,}")
print(f"Rows after within-source clause consolidation: {rows_after_source_consolidation:,}")
print(f"Cross-source duplicate clauses removed: {rows_after_source_consolidation - len(FearValResearchData_combined):,}")
print(f"Final research population: {len(FearValResearchData_combined):,} unique clauses")
print(pd.DataFrame(repair_summary).T.to_string())
print(FearValResearchData_combined["file name"].value_counts().to_string())
FearValResearchData_combined.head()
```

    Created FearValResearchData_combined.xlsx
    Created FearValResearchData_combined_woArt.xlsx
    Article elements removed from second version: 17 across 17 clauses
    Header source: FearValResearchData_02_DOwiET.xlsx (123 columns)
    Input rows: 433
    Rows after within-source clause consolidation: 129
    Cross-source duplicate clauses removed: 2
    Final research population: 127 unique clauses
                                     higher complement nodes placed  cleared single-word element-2  highest complement element
    FearValResearchData_03_DOwoET                                 0                              0                           1
    FearValResearchData_02_DOwiET                                67                              0                           8
    FearValResearchData_04_DOasPreO                               0                              0                           1
    FearValResearchData_01_MN                                    74                             10                           5
    file name
    FearValResearchData_01_MN          52
    FearValResearchData_02_DOwiET      45
    FearValResearchData_03_DOwoET      18
    FearValResearchData_04_DOasPreO    12





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file name</th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>...</th>
      <th>word H Cmpl-element-8 part of speech</th>
      <th>word H Cmpl-element-8 number</th>
      <th>word H Cmpl-element-8 person</th>
      <th>word H Cmpl-element-8 gender</th>
      <th>word H Cmpl-element-8 stem</th>
      <th>word H Cmpl-element-8 tense</th>
      <th>word H lex Cmpl-element-8</th>
      <th>word H lexUTF8 Cmpl-element-8</th>
      <th>word G text Cmpl-element-8</th>
      <th>word G lex Cmpl-element-8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>1</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>439829</td>
      <td>אִמֹּ֤ו וְאָבִיו֙ תִּירָ֔אוּ</td>
      <td>688881</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>2</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>30</td>
      <td>439925</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>689186</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>3</td>
      <td>Leviticus</td>
      <td>26</td>
      <td>2</td>
      <td>440669</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>691449</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>4</td>
      <td>Deuteronomium</td>
      <td>25</td>
      <td>18</td>
      <td>447594</td>
      <td>וְלֹ֥א יָרֵ֖א אֱלֹהִֽים׃</td>
      <td>712454</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>5</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>7</td>
      <td>467059</td>
      <td>וַיִּֽירְא֖וּ אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>770760</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 123 columns</p>
</div>



## Adding Annotations to Research data


```python
from pathlib import Path

import pandas as pd
from openpyxl.styles import PatternFill
from openpyxl.utils import get_column_letter

data_export_dir = Path.cwd() / "data_export"
combined_files = [
    data_export_dir / "FearValResearchData_combined.xlsx",
    data_export_dir / "FearValResearchData_combined_woArt.xlsx",
]
annotation_file = Path.cwd() / "TheValenceOfFear_HIPHIL_o.glanz_v20260814.xlsx"
annotation_sheet = "2026_combined_HIPHIL"

annotation_columns = [
    "complement type",
    "speaker",
    "BHS_negation",
    "GT_negation",
    "topic",
    "suffix",
    "interpretation",
    "demo",
    "deviation",
    "usable",
    "interesting cases",
]

annotations = pd.read_excel(annotation_file, sheet_name=annotation_sheet)
required_annotation_columns = ["clause_node", *annotation_columns]
missing_columns = [
    column for column in required_annotation_columns
    if column not in annotations.columns
]
if missing_columns:
    raise KeyError(f"Missing annotation columns: {missing_columns}")

if annotations["clause_node"].isna().any():
    raise ValueError("Blank clause_node values found in the annotation workbook")
if annotations["clause_node"].duplicated().any():
    duplicate_nodes = (
        annotations.loc[
            annotations["clause_node"].duplicated(keep=False),
            "clause_node",
        ]
        .drop_duplicates()
        .tolist()
    )
    raise ValueError(f"Duplicate clause_node values in annotations: {duplicate_nodes}")

annotations_for_merge = annotations[required_annotation_columns].copy()
element_color_pairs = [
    ("9DC3E6", "DDEBF7"),  # blue
    ("F4B183", "FCE4D6"),  # orange
    ("A9D18E", "E2F0D9"),  # green
    ("B4A7D6", "E4DFEC"),  # purple
    ("F4D35E", "FFF2CC"),  # yellow
    ("80CBC4", "DDEBF7"),  # teal
    ("E6B8C7", "F4CCCC"),  # rose
    ("BFBFBF", "E7E6E6"),  # gray
]

def append_annotations(combined_file):
    """Append the selected research annotations and preserve workbook formatting."""
    combined = pd.read_excel(combined_file)

    # Make the cell safe to rerun by replacing existing annotation columns.
    existing_annotation_columns = [
        column for column in annotation_columns if column in combined.columns
    ]
    if existing_annotation_columns:
        combined = combined.drop(columns=existing_annotation_columns)

    combined = (
        combined.merge(
            annotations_for_merge,
            left_on="clause #",
            right_on="clause_node",
            how="left",
            validate="one_to_one",
            sort=False,
        )
        .drop(columns="clause_node")
    )
    unmatched_clause_numbers = combined.loc[
        ~combined["clause #"].isin(annotations["clause_node"]),
        "clause #",
    ].tolist()

    with pd.ExcelWriter(combined_file, engine="openpyxl") as writer:
        combined.to_excel(writer, sheet_name="Sheet1", index=False)
        worksheet = writer.book["Sheet1"]
        worksheet.freeze_panes = "A2"
        worksheet.auto_filter.ref = worksheet.dimensions

        element_numbers = sorted({
            int(column_name.split("Cmpl-element-", 1)[1].split()[0])
            for column_name in combined.columns
            if "Cmpl-element-" in column_name
        })
        for element_number in element_numbers:
            header_color, body_color = element_color_pairs[
                (element_number - 1) % len(element_color_pairs)
            ]
            header_fill = PatternFill("solid", fgColor=header_color)
            body_fill = PatternFill("solid", fgColor=body_color)
            element_marker = f"Cmpl-element-{element_number}"
            element_column_numbers = [
                column_number
                for column_number, column_name in enumerate(combined.columns, start=1)
                if element_marker in column_name
            ]
            for column_number in element_column_numbers:
                worksheet.cell(1, column_number).fill = header_fill
                for row_number in range(2, worksheet.max_row + 1):
                    worksheet.cell(row_number, column_number).fill = body_fill

        for column_number, column_name in enumerate(combined.columns, start=1):
            sample = combined[column_name].dropna().astype(str)
            content_width = sample.str.len().max() if not sample.empty else 0
            worksheet.column_dimensions[get_column_letter(column_number)].width = min(
                40, max(3, content_width + 2)
            )

    return combined, unmatched_clause_numbers

annotation_results = {}
for combined_file in combined_files:
    combined_data, unmatched_clause_numbers = append_annotations(combined_file)
    annotation_results[combined_file.name] = {
        "rows": len(combined_data),
        "matched": len(combined_data) - len(unmatched_clause_numbers),
        "unmatched": unmatched_clause_numbers,
    }
    print(f"Updated {combined_file.name}")
    print(f"Appended {len(annotation_columns)} annotation columns")
    print(f"Matched {len(combined_data) - len(unmatched_clause_numbers):,} of {len(combined_data):,} clauses")
    print(f"Unmatched clause numbers: {unmatched_clause_numbers}")

FearValResearchData_combined = pd.read_excel(combined_files[0])
FearValResearchData_combined_woArt = pd.read_excel(combined_files[1])
FearValResearchData_combined_woArt.head()

```

    Updated FearValResearchData_combined.xlsx
    Appended 11 annotation columns
    Matched 126 of 127 clauses
    Unmatched clause numbers: [497943]
    Updated FearValResearchData_combined_woArt.xlsx
    Appended 11 annotation columns
    Matched 126 of 127 clauses
    Unmatched clause numbers: [497943]





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>file name</th>
      <th>order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause #</th>
      <th>verse_text</th>
      <th>phrase # Pred</th>
      <th>function Pred</th>
      <th>phrase type Pred</th>
      <th>...</th>
      <th>speaker</th>
      <th>BHS_negation</th>
      <th>GT_negation</th>
      <th>topic</th>
      <th>suffix</th>
      <th>interpretation</th>
      <th>demo</th>
      <th>deviation</th>
      <th>usable</th>
      <th>interesting cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>1</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>439829</td>
      <td>אִמֹּ֤ו וְאָבִיו֙ תִּירָ֔אוּ</td>
      <td>688881</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>YHWH</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Parents</td>
      <td>NaN</td>
      <td>respect</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>2</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>30</td>
      <td>439925</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>689186</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>YHWH</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sanctuary</td>
      <td>NaN</td>
      <td>respect</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>3</td>
      <td>Leviticus</td>
      <td>26</td>
      <td>2</td>
      <td>440669</td>
      <td>וּמִקְדָּשִׁ֖י תִּירָ֑אוּ</td>
      <td>691449</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>YHWH</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Sanctuary</td>
      <td>NaN</td>
      <td>respect</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>4</td>
      <td>Deuteronomium</td>
      <td>25</td>
      <td>18</td>
      <td>447594</td>
      <td>וְלֹ֥א יָרֵ֖א אֱלֹהִֽים׃</td>
      <td>712454</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>LO</td>
      <td>οὐκ</td>
      <td>God</td>
      <td>NaN</td>
      <td>respect</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>FearValResearchData_03_DOwoET</td>
      <td>5</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>7</td>
      <td>467059</td>
      <td>וַיִּֽירְא֖וּ אֱלֹהִ֥ים אֲחֵרִֽים׃</td>
      <td>770760</td>
      <td>Pred</td>
      <td>VP</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>God</td>
      <td>NaN</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 134 columns</p>
</div>



## data generation


```python
from pathlib import Path

# Use the 127 curated records from the original 2026_combined sheet and
# append Tov LXX surface-text and lexeme features for every Hebrew word node.

original_file = Path.cwd() / "TheValenceOfFear_BRILL_o.glanz_v20260120.xlsx"
alignment_file = Path.cwd() / "data_export" / "FearValDataExport_all.tsv"
HIPHIL_combined = pd.read_excel(original_file, sheet_name="2026_combined")
alignment = pd.read_csv(alignment_file, delimiter="\t", encoding="utf-16")

node_columns = ["NODE4", "NODE6", "NODE7"]
tov_source_columns = ["tov_LXX_text6", "tov_LXX_lex6"]

# NODE6 enumerates the Hebrew word nodes in the alignment export. Each node
# has at most one Tov LXX text/lexeme pair, so it can serve as the lookup for
# the predicate (NODE4) and both complement-word positions (NODE6/NODE7).
tov_lookup = (
    alignment[["NODE6", *tov_source_columns]]
    .dropna(subset=["NODE6"])
    .drop_duplicates()
)
conflicting_nodes = tov_lookup.groupby("NODE6").size().gt(1)
if conflicting_nodes.any():
    raise ValueError(
        f"Conflicting Tov alignments for {conflicting_nodes.sum()} Hebrew nodes"
    )
tov_lookup = tov_lookup.set_index("NODE6")

for node_column in node_columns:
    HIPHIL_combined[f"{node_column}_tov_LXX_text"] = (
        HIPHIL_combined[node_column].map(tov_lookup["tov_LXX_text6"])
    )
    HIPHIL_combined[f"{node_column}_tov_LXX_lex"] = (
        HIPHIL_combined[node_column].map(tov_lookup["tov_LXX_lex6"])
    )
original_columns = pd.read_excel(
    original_file, sheet_name="2026_combined", nrows=0
).columns.tolist()
output_columns = []
tov_columns_after = {
    "lex_utf84": ["NODE4_tov_LXX_text", "NODE4_tov_LXX_lex"],
    "lex_utf86": ["NODE6_tov_LXX_text", "NODE6_tov_LXX_lex"],
    "lex_utf87": ["NODE7_tov_LXX_text", "NODE7_tov_LXX_lex"],
}
for column in original_columns:
    output_columns.append(column)
    output_columns.extend(tov_columns_after.get(column, []))
HIPHIL_combined = HIPHIL_combined[output_columns]

output_file = Path.cwd() / "TheValenceOfFear_HIPHIL_o.glanz_v20260813.xlsx"
sheet_name = "2026_combined"

with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
    HIPHIL_combined.to_excel(writer, sheet_name=sheet_name, index=False)

    worksheet = writer.book[sheet_name]
    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions

    # Plain, machine-readable formatting with visible headers.
    from openpyxl.styles import Alignment, Font, PatternFill
    header_fill = PatternFill("solid", fgColor="1F4E78")
    for cell in worksheet[1]:
        cell.fill = header_fill
        cell.font = Font(color="FFFFFF", bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center")

    width_by_column = {
        "A": 10, "B": 16, "C": 10, "D": 8, "E": 12, "F": 12,
        "G": 22, "H": 14, "I": 12, "J": 10, "K": 18, "L": 16,
        "M": 18, "N": 16, "O": 10, "P": 10, "Q": 10, "R": 10,
        "S": 20, "T": 24, "U": 22,
    }
    for column_letter, width in width_by_column.items():
        worksheet.column_dimensions[column_letter].width = width

print(f"Created {output_file.name}")
print(f"{len(HIPHIL_combined):,} curated valence records exported")
for node_column in node_columns:
    text_count = HIPHIL_combined[f"{node_column}_tov_LXX_text"].notna().sum()
    lex_count = HIPHIL_combined[f"{node_column}_tov_LXX_lex"].notna().sum()
    print(f"{node_column}: {text_count:,} Tov texts; {lex_count:,} Tov lexemes")
HIPHIL_combined.head()
```

    Created TheValenceOfFear_HIPHIL_o.glanz_v20260813.xlsx
    127 curated valence records exported
    NODE4: 127 Tov texts; 125 Tov lexemes
    NODE6: 58 Tov texts; 36 Tov lexemes
    NODE7: 118 Tov texts; 103 Tov lexemes





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>R</th>
      <th>S1</th>
      <th>S2</th>
      <th>S3</th>
      <th>NODE1</th>
      <th>TYPE1</th>
      <th>TEXT1</th>
      <th>book1</th>
      <th>chapter1</th>
      <th>verse1</th>
      <th>...</th>
      <th>vt7</th>
      <th>complement type</th>
      <th>speaker</th>
      <th>negation</th>
      <th>topic</th>
      <th>suffix</th>
      <th>interpretation</th>
      <th>demo</th>
      <th>deviation</th>
      <th>usable</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2101</td>
      <td>Exodus</td>
      <td>1</td>
      <td>17</td>
      <td>1415938</td>
      <td>verse</td>
      <td>וַתִּירֶ֤אןָ הַֽמְיַלְּדֹת֙ אֶת־הָ֣אֱלֹהִ֔ים ו...</td>
      <td>Exodus</td>
      <td>1</td>
      <td>17</td>
      <td>...</td>
      <td>NaN</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>God</td>
      <td>NaN</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2102</td>
      <td>Exodus</td>
      <td>1</td>
      <td>21</td>
      <td>1415942</td>
      <td>verse</td>
      <td>וַיְהִ֕י כִּֽי־יָֽרְא֥וּ הַֽמְיַלְּדֹ֖ת אֶת־הָ...</td>
      <td>Exodus</td>
      <td>1</td>
      <td>21</td>
      <td>...</td>
      <td>NaN</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>God</td>
      <td>NaN</td>
      <td>respect</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1101</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>1416161</td>
      <td>verse</td>
      <td>וְאַתָּ֖ה וַעֲבָדֶ֑יךָ יָדַ֕עְתִּי כִּ֚י טֶ֣רֶ...</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>...</td>
      <td>NaN</td>
      <td>MN_complement</td>
      <td>Moses</td>
      <td>NaN</td>
      <td>Face of JHWH</td>
      <td>NaN</td>
      <td>respect/afraid</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2103</td>
      <td>Exodus</td>
      <td>14</td>
      <td>31</td>
      <td>1416309</td>
      <td>verse</td>
      <td>וַיַּ֨רְא יִשְׂרָאֵ֜ל אֶת־הַיָּ֣ד הַגְּדֹלָ֗ה ...</td>
      <td>Exodus</td>
      <td>14</td>
      <td>31</td>
      <td>...</td>
      <td>NaN</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>YHWH</td>
      <td>NaN</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2301</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>1417673</td>
      <td>verse</td>
      <td>אִ֣ישׁ אִמֹּ֤ו וְאָבִיו֙ תִּירָ֔אוּ וְאֶת־שַׁב...</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>3</td>
      <td>...</td>
      <td>NaN</td>
      <td>DO_complement</td>
      <td>YHWH</td>
      <td>NaN</td>
      <td>Parents</td>
      <td>NaN</td>
      <td>respect</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 80 columns</p>
</div>



## Research Data Loading


```python
from pathlib import Path

# Load the manually cleaned research dataset.
workbook_file = Path.cwd() / "TheValenceOfFear_HIPHIL_o.glanz_v20260813_manual.xlsx"
df = pd.read_excel(workbook_file, sheet_name="2026_combined_HIPHIL")
pd.set_option("display.max_columns", None)

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>orig_order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>verse_node</th>
      <th>verse_text_BHS</th>
      <th>clause_node</th>
      <th>clause_text_BHS</th>
      <th>clause_text_GT</th>
      <th>phrase1_node</th>
      <th>phrase1_text_BHS</th>
      <th>phrase1_text_GT</th>
      <th>phrase1_function</th>
      <th>phrase1_type</th>
      <th>phrase1word1_node</th>
      <th>phrase1word1_text_BHS</th>
      <th>phrase1word1_text_GT</th>
      <th>phrase1word1_text_BHS_cons</th>
      <th>phrase1word1_text_BHS_gn</th>
      <th>phrase1word1_BHS_nu</th>
      <th>phrase1word1_BHS_ps</th>
      <th>phrase1word1_BHS_sp</th>
      <th>phrase1word1_BHS_vs</th>
      <th>phrase1word1_BHS_vt</th>
      <th>phrase1word1_text_BHS_lex</th>
      <th>phrase1word1_text_BHS_lexUTF8</th>
      <th>phrase1word1_text_GT_lexSBL</th>
      <th>phrase1word1_text_GT_lexUTF8</th>
      <th>phrase2_node</th>
      <th>phrase2_text_BHS</th>
      <th>phrase2_text_GT</th>
      <th>phrase2_function</th>
      <th>phrase2_function.1</th>
      <th>phrase2word1_node</th>
      <th>phrase2word1_prep_BHS_text</th>
      <th>phrase2word1_prep_BHS_lex</th>
      <th>phrase2word1_prep_BHS_lexUTF8</th>
      <th>phrase2word1_prep_GT_lexUTF8</th>
      <th>phrase2word2_GT_text</th>
      <th>phrase2word2_GT_lexUTF8</th>
      <th>phrase2word3_node</th>
      <th>phrase2word3_article_BHS_text</th>
      <th>phrase2word3_article_GT_text</th>
      <th>phrase2word3_prep_BHS_lex</th>
      <th>phrase2word3_prep_BHS_lexUTF8</th>
      <th>phrase2word3_prep_GT_lexUTF8</th>
      <th>phrase2word4_possesive_BHS_lexUTF8</th>
      <th>phrase2word4_possesive_GT_lexUTF8</th>
      <th>phrase2word4_adjective_BHS_lexUTF8</th>
      <th>phrase2word4_adjective_GT_lexUTF8</th>
      <th>phrase2word4_extensions_BHS_lexUTF8</th>
      <th>phrase2word4_extensions_GT_lexUTF8</th>
      <th>complement type</th>
      <th>speaker</th>
      <th>BHS_negation</th>
      <th>GT_negation</th>
      <th>topic</th>
      <th>suffix</th>
      <th>interpretation</th>
      <th>demo</th>
      <th>deviation</th>
      <th>usable</th>
      <th>interesting cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>Deuteronomium</td>
      <td>3</td>
      <td>22</td>
      <td>1419387</td>
      <td>לֹ֖א תִּֽירָא֑וּם כִּ֚י יְהוָ֣ה אֱלֹֽהֵיכֶ֔ם ה...</td>
      <td>445283</td>
      <td>לֹ֖א תִּֽירָא֑וּם</td>
      <td>οὐ φοβηθήσεσθε ὅτι κύριος ὁ θεὸς ἡμῶν αὐτὸς πο...</td>
      <td>705411</td>
      <td>תִּֽירָא֑וּם</td>
      <td>φοβηθήσεσθε</td>
      <td>PreO</td>
      <td>VP</td>
      <td>95000</td>
      <td>תִּֽירָא֑וּם</td>
      <td>φοβηθήσεσθε</td>
      <td>TJR&gt;WM</td>
      <td>m</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>phobeō</td>
      <td>φοβέω</td>
      <td>705411</td>
      <td>תִּֽירָא֑וּם</td>
      <td>absent</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>95000</td>
      <td>NaN</td>
      <td>absent</td>
      <td>prs</td>
      <td>prs</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>DO_complement</td>
      <td>Joshua</td>
      <td>LO</td>
      <td>οὐ</td>
      <td>Canaanites_suffix</td>
      <td>y</td>
      <td>afraid with PreO as short from of Jara + MN</td>
      <td>y</td>
      <td>x</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106</td>
      <td>Psalmi</td>
      <td>72</td>
      <td>5</td>
      <td>1430654</td>
      <td>יִֽירָא֥וּךָ עִם־שָׁ֑מֶשׁ וְלִפְנֵ֥י יָ֝רֵ֗חַ ...</td>
      <td>493723</td>
      <td>יִֽירָא֥וּךָ עִם־שָׁ֑מֶשׁ וְלִפְנֵ֥י יָ֝רֵ֗חַ ...</td>
      <td>καὶ συμπαραμενεῖ τῷ ἡλίῳ καὶ πρὸ τῆς σελήνης γ...</td>
      <td>843694</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>καὶ|συμπαραμενεῖ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>322351</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>καὶ|συμπαραμενεῖ</td>
      <td>JJR&gt;WK</td>
      <td>m</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>symparamenō</td>
      <td>συμπαραμένω</td>
      <td>843694</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>absent</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>322351</td>
      <td>NaN</td>
      <td>absent</td>
      <td>prs</td>
      <td>prs</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>YHWH_suffix</td>
      <td>y</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>115</td>
      <td>Iob</td>
      <td>9</td>
      <td>35</td>
      <td>1432281</td>
      <td>אַֽ֭דַבְּרָה וְלֹ֣א אִירָאֶ֑נּוּ כִּ֥י לֹא־כֵ֥...</td>
      <td>498186</td>
      <td>וְלֹ֣א אִירָאֶ֑נּוּ</td>
      <td>καὶ οὐ μὴ φοβηθῶ ἀλλὰ λαλήσω οὐ γὰρ οὕτω συνεπ...</td>
      <td>854986</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>φοβηθῶ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>338580</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>φοβηθῶ</td>
      <td>&gt;JR&gt;NW</td>
      <td>unknown</td>
      <td>sg</td>
      <td>p1</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>phobeō</td>
      <td>φοβέω</td>
      <td>854986</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>absent</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>338580</td>
      <td>NaN</td>
      <td>absent</td>
      <td>prs</td>
      <td>prs</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>LO</td>
      <td>οὐ μή</td>
      <td>YHWH_suffix</td>
      <td>y</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>Jeremia</td>
      <td>10</td>
      <td>7</td>
      <td>1426112</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙ מֶ֣לֶךְ הַגֹּויִ֔ם כִּ֥י ...</td>
      <td>475070</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙</td>
      <td>TC:missing</td>
      <td>792664</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>missing</td>
      <td>PreO</td>
      <td>VP</td>
      <td>240005</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>missing</td>
      <td>JR&gt;K</td>
      <td>m</td>
      <td>sg</td>
      <td>p3</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>missing</td>
      <td>missing</td>
      <td>792664</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>TC:missing</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>TC:missing</td>
      <td>TC:missing</td>
      <td>240005</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>prs</td>
      <td>prs</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>LO</td>
      <td>TC:missing</td>
      <td>YHWH_suffix</td>
      <td>y</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>81</td>
      <td>Jeremia</td>
      <td>42</td>
      <td>11</td>
      <td>1426890</td>
      <td>אַל־תִּֽירְא֗וּ מִפְּנֵי֙ מֶ֣לֶךְ בָּבֶ֔ל אֲשׁ...</td>
      <td>478597</td>
      <td>אַל־תִּֽירְא֤וּ מִמֶּ֨נּוּ֙</td>
      <td>μὴ φοβηθῆτε ἀπὸ προσώπου βασιλέως Βαβυλῶνος οὗ...</td>
      <td>802878</td>
      <td>תִּֽירְא֤וּ</td>
      <td>φοβηθῆτε</td>
      <td>Pred</td>
      <td>VP</td>
      <td>258043</td>
      <td>תִּֽירְא֤וּ</td>
      <td>φοβηθῆτε</td>
      <td>TJR&gt;W</td>
      <td>m</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>phobeō</td>
      <td>φοβέω</td>
      <td>802879</td>
      <td>מִמֶּ֨נּוּ֙</td>
      <td>TC:missing</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>258044.0</td>
      <td>מִמֶּ֨נּוּ֙</td>
      <td>MN</td>
      <td>מן</td>
      <td>TC:missing</td>
      <td>absent</td>
      <td>TC:missing</td>
      <td>258044</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>prs</td>
      <td>prs</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>MN_complement</td>
      <td>YHWH</td>
      <td>AL</td>
      <td>μή</td>
      <td>King of Babel_suffix</td>
      <td>NaN</td>
      <td>afraid</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Use a clearer name for the source ordering column.
df = df.rename(columns={"orig_order": "original_order"})

df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original_order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>verse_node</th>
      <th>verse_text_BHS</th>
      <th>clause_node</th>
      <th>clause_text_BHS</th>
      <th>clause_text_GT</th>
      <th>phrase1_node</th>
      <th>phrase1_text_BHS</th>
      <th>phrase1_text_GT</th>
      <th>phrase1_function</th>
      <th>phrase1_type</th>
      <th>phrase1word1_node</th>
      <th>phrase1word1_text_BHS</th>
      <th>phrase1word1_text_GT</th>
      <th>phrase1word1_text_BHS_cons</th>
      <th>phrase1word1_text_BHS_gn</th>
      <th>phrase1word1_BHS_nu</th>
      <th>phrase1word1_BHS_ps</th>
      <th>phrase1word1_BHS_sp</th>
      <th>phrase1word1_BHS_vs</th>
      <th>phrase1word1_BHS_vt</th>
      <th>phrase1word1_text_BHS_lex</th>
      <th>phrase1word1_text_BHS_lexUTF8</th>
      <th>phrase1word1_text_GT_lexSBL</th>
      <th>phrase1word1_text_GT_lexUTF8</th>
      <th>phrase2_node</th>
      <th>phrase2_text_BHS</th>
      <th>phrase2_text_GT</th>
      <th>phrase2_function</th>
      <th>phrase2_function.1</th>
      <th>phrase2word1_node</th>
      <th>phrase2word1_prep_BHS_text</th>
      <th>phrase2word1_prep_BHS_lex</th>
      <th>phrase2word1_prep_BHS_lexUTF8</th>
      <th>phrase2word1_prep_GT_lexUTF8</th>
      <th>phrase2word2_GT_text</th>
      <th>phrase2word2_GT_lexUTF8</th>
      <th>phrase2word3_node</th>
      <th>phrase2word3_article_BHS_text</th>
      <th>phrase2word3_article_GT_text</th>
      <th>phrase2word3_prep_BHS_lex</th>
      <th>phrase2word3_prep_BHS_lexUTF8</th>
      <th>phrase2word3_prep_GT_lexUTF8</th>
      <th>phrase2word4_possesive_BHS_lexUTF8</th>
      <th>phrase2word4_possesive_GT_lexUTF8</th>
      <th>phrase2word4_adjective_BHS_lexUTF8</th>
      <th>phrase2word4_adjective_GT_lexUTF8</th>
      <th>phrase2word4_extensions_BHS_lexUTF8</th>
      <th>phrase2word4_extensions_GT_lexUTF8</th>
      <th>complement type</th>
      <th>speaker</th>
      <th>BHS_negation</th>
      <th>GT_negation</th>
      <th>topic</th>
      <th>suffix</th>
      <th>interpretation</th>
      <th>demo</th>
      <th>deviation</th>
      <th>usable</th>
      <th>interesting cases</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>19</td>
      <td>Deuteronomium</td>
      <td>3</td>
      <td>22</td>
      <td>1419387</td>
      <td>לֹ֖א תִּֽירָא֑וּם כִּ֚י יְהוָ֣ה אֱלֹֽהֵיכֶ֔ם ה...</td>
      <td>445283</td>
      <td>לֹ֖א תִּֽירָא֑וּם</td>
      <td>οὐ φοβηθήσεσθε ὅτι κύριος ὁ θεὸς ἡμῶν αὐτὸς πο...</td>
      <td>705411</td>
      <td>תִּֽירָא֑וּם</td>
      <td>φοβηθήσεσθε</td>
      <td>PreO</td>
      <td>VP</td>
      <td>95000</td>
      <td>תִּֽירָא֑וּם</td>
      <td>φοβηθήσεσθε</td>
      <td>TJR&gt;WM</td>
      <td>m</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>phobeō</td>
      <td>φοβέω</td>
      <td>705411</td>
      <td>תִּֽירָא֑וּם</td>
      <td>absent</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>95000</td>
      <td>NaN</td>
      <td>absent</td>
      <td>prs</td>
      <td>prs</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>DO_complement</td>
      <td>Joshua</td>
      <td>LO</td>
      <td>οὐ</td>
      <td>Canaanites_suffix</td>
      <td>y</td>
      <td>afraid with PreO as short from of Jara + MN</td>
      <td>y</td>
      <td>x</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>106</td>
      <td>Psalmi</td>
      <td>72</td>
      <td>5</td>
      <td>1430654</td>
      <td>יִֽירָא֥וּךָ עִם־שָׁ֑מֶשׁ וְלִפְנֵ֥י יָ֝רֵ֗חַ ...</td>
      <td>493723</td>
      <td>יִֽירָא֥וּךָ עִם־שָׁ֑מֶשׁ וְלִפְנֵ֥י יָ֝רֵ֗חַ ...</td>
      <td>καὶ συμπαραμενεῖ τῷ ἡλίῳ καὶ πρὸ τῆς σελήνης γ...</td>
      <td>843694</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>καὶ|συμπαραμενεῖ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>322351</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>καὶ|συμπαραμενεῖ</td>
      <td>JJR&gt;WK</td>
      <td>m</td>
      <td>pl</td>
      <td>p3</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>symparamenō</td>
      <td>συμπαραμένω</td>
      <td>843694</td>
      <td>יִֽירָא֥וּךָ</td>
      <td>absent</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>322351</td>
      <td>NaN</td>
      <td>absent</td>
      <td>prs</td>
      <td>prs</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>YHWH_suffix</td>
      <td>y</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>115</td>
      <td>Iob</td>
      <td>9</td>
      <td>35</td>
      <td>1432281</td>
      <td>אַֽ֭דַבְּרָה וְלֹ֣א אִירָאֶ֑נּוּ כִּ֥י לֹא־כֵ֥...</td>
      <td>498186</td>
      <td>וְלֹ֣א אִירָאֶ֑נּוּ</td>
      <td>καὶ οὐ μὴ φοβηθῶ ἀλλὰ λαλήσω οὐ γὰρ οὕτω συνεπ...</td>
      <td>854986</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>φοβηθῶ</td>
      <td>PreO</td>
      <td>VP</td>
      <td>338580</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>φοβηθῶ</td>
      <td>&gt;JR&gt;NW</td>
      <td>unknown</td>
      <td>sg</td>
      <td>p1</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>phobeō</td>
      <td>φοβέω</td>
      <td>854986</td>
      <td>אִירָאֶ֑נּוּ</td>
      <td>absent</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>absent</td>
      <td>338580</td>
      <td>NaN</td>
      <td>absent</td>
      <td>prs</td>
      <td>prs</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>NaN</td>
      <td>absent</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>LO</td>
      <td>οὐ μή</td>
      <td>YHWH_suffix</td>
      <td>y</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>77</td>
      <td>Jeremia</td>
      <td>10</td>
      <td>7</td>
      <td>1426112</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙ מֶ֣לֶךְ הַגֹּויִ֔ם כִּ֥י ...</td>
      <td>475070</td>
      <td>מִ֣י לֹ֤א יִֽרָאֲךָ֙</td>
      <td>TC:missing</td>
      <td>792664</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>missing</td>
      <td>PreO</td>
      <td>VP</td>
      <td>240005</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>missing</td>
      <td>JR&gt;K</td>
      <td>m</td>
      <td>sg</td>
      <td>p3</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>missing</td>
      <td>missing</td>
      <td>792664</td>
      <td>יִֽרָאֲךָ֙</td>
      <td>TC:missing</td>
      <td>PreO</td>
      <td>VP</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>TC:missing</td>
      <td>TC:missing</td>
      <td>240005</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>prs</td>
      <td>prs</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>DO_complement</td>
      <td>NaN</td>
      <td>LO</td>
      <td>TC:missing</td>
      <td>YHWH_suffix</td>
      <td>y</td>
      <td>respect</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>81</td>
      <td>Jeremia</td>
      <td>42</td>
      <td>11</td>
      <td>1426890</td>
      <td>אַל־תִּֽירְא֗וּ מִפְּנֵי֙ מֶ֣לֶךְ בָּבֶ֔ל אֲשׁ...</td>
      <td>478597</td>
      <td>אַל־תִּֽירְא֤וּ מִמֶּ֨נּוּ֙</td>
      <td>μὴ φοβηθῆτε ἀπὸ προσώπου βασιλέως Βαβυλῶνος οὗ...</td>
      <td>802878</td>
      <td>תִּֽירְא֤וּ</td>
      <td>φοβηθῆτε</td>
      <td>Pred</td>
      <td>VP</td>
      <td>258043</td>
      <td>תִּֽירְא֤וּ</td>
      <td>φοβηθῆτε</td>
      <td>TJR&gt;W</td>
      <td>m</td>
      <td>pl</td>
      <td>p2</td>
      <td>verb</td>
      <td>qal</td>
      <td>impf</td>
      <td>JR&gt;[</td>
      <td>ירא</td>
      <td>phobeō</td>
      <td>φοβέω</td>
      <td>802879</td>
      <td>מִמֶּ֨נּוּ֙</td>
      <td>TC:missing</td>
      <td>Cmpl</td>
      <td>PP</td>
      <td>258044.0</td>
      <td>מִמֶּ֨נּוּ֙</td>
      <td>MN</td>
      <td>מן</td>
      <td>TC:missing</td>
      <td>absent</td>
      <td>TC:missing</td>
      <td>258044</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>prs</td>
      <td>prs</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>NaN</td>
      <td>TC:missing</td>
      <td>MN_complement</td>
      <td>YHWH</td>
      <td>AL</td>
      <td>μή</td>
      <td>King of Babel_suffix</td>
      <td>NaN</td>
      <td>afraid</td>
      <td>y</td>
      <td>NaN</td>
      <td>y</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



## Add word-level Greek parallels to the manually formatted HIPHIL dataset


```python
from copy import copy
from pathlib import Path

import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

manual_file = Path.cwd() / "TheValenceOfFear_HIPHIL_o.glanz_v20260814.xlsx"
alignment_file = Path.cwd() / "data_export" / "FearValDataExport_all.tsv"
sheet_name = "2026_combined_HIPHIL"

alignment = pd.read_csv(alignment_file, delimiter="\t", encoding="utf-16")
source_columns = [
    "macula_BH_lxx_lex_16",
    "macula_BH_lxx_lex_26",
    "macula_BH_lxx_lex_36",
    "macula_BH_lxx_text6",
]
feature_labels = [
    "macula_BH_lxx_lex_1",
    "macula_BH_lxx_lex_2",
    "macula_BH_lxx_lex_3",
    "macula_BH_lxx_text",
]

# Every TSV word node must have at most one distinct set of Greek values.
node_alignment = alignment[["NODE6"] + source_columns].drop_duplicates()
conflicts = node_alignment.groupby("NODE6").size()
if conflicts.gt(1).any():
    raise ValueError("Conflicting Greek alignments found for one or more word nodes")
node_alignment = node_alignment.set_index("NODE6")

workbook = load_workbook(manual_file)
worksheet = workbook[sheet_name]
headers = {cell.value: cell.column for cell in worksheet[1] if cell.value is not None}
node_columns = ["NODE4", "NODE6", "NODE7"]
missing = set(node_columns).difference(headers)
if missing:
    raise KeyError(f"Missing node columns: {sorted(missing)}")

new_headers = [
    f"{node}_{feature}"
    for node in node_columns
    for feature in feature_labels
]
start_column = headers.get(new_headers[0], worksheet.max_column + 1)

# On first execution, copy only the formatting of an existing neutral
# twelve-column block. Existing cells, filters, freezes, fonts and fills
# remain untouched. Subsequent executions overwrite only the Greek values.
if new_headers[0] not in headers:
    for offset in range(12):
        source_column = 17 + offset  # Q:AB
        target_column = start_column + offset
        source_letter = get_column_letter(source_column)
        target_letter = get_column_letter(target_column)
        worksheet.column_dimensions[target_letter].width = 28
        for row in range(1, worksheet.max_row + 1):
            source_cell = worksheet.cell(row, source_column)
            target_cell = worksheet.cell(row, target_column)
            target_cell._style = copy(source_cell._style)
            target_cell.number_format = source_cell.number_format
            target_cell.alignment = copy(source_cell.alignment)
            target_cell.protection = copy(source_cell.protection)

for offset, header in enumerate(new_headers):
    worksheet.cell(1, start_column + offset, header)

for row in range(2, worksheet.max_row + 1):
    output = []
    for node_column in node_columns:
        node = worksheet.cell(row, headers[node_column]).value
        if node is None or node not in node_alignment.index:
            output.extend([None] * 4)
        else:
            output.extend(node_alignment.loc[node, source_columns].tolist())
    for offset, value in enumerate(output):
        worksheet.cell(row, start_column + offset, value)

last_column = get_column_letter(start_column + len(new_headers) - 1)
worksheet.auto_filter.ref = f"A1:{last_column}{worksheet.max_row}"
for table in worksheet.tables.values():
    table.ref = f"A1:{last_column}{worksheet.max_row}"
    if table.autoFilter is not None:
        table.autoFilter.ref = table.ref

workbook.save(manual_file)
print(f"Updated {manual_file.name}: {worksheet.max_row - 1} records, {len(new_headers)} Greek columns")
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    Cell In[25], line 39
         37 missing = set(node_columns).difference(headers)
         38 if missing:
    ---> 39     raise KeyError(f"Missing node columns: {sorted(missing)}")
         41 new_headers = [
         42     f"{node}_{feature}"
         43     for node in node_columns
         44     for feature in feature_labels
         45 ]
         46 start_column = headers.get(new_headers[0], worksheet.max_column + 1)


    KeyError: "Missing node columns: ['NODE4', 'NODE6', 'NODE7']"


# Distribution of Greek verbs translating Hebrew ירא


```python
import matplotlib.pyplot as plt

hebrew_verb = "ירא"
hebrew_lexeme_column = "phrase1word1_text_BHS_lexUTF8"
greek_lexeme_column = "phrase1word1_text_GT_lexUTF8"

Greek_verbs_for_yare = (
    df.loc[df[hebrew_lexeme_column].eq(hebrew_verb), greek_lexeme_column]
    .fillna("[no aligned Greek lexeme]")
    .value_counts()
    .rename_axis("Greek verb")
    .rename("frequency")
)

Greek_verbs_for_yare.to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>Greek verb</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>φοβέω</th>
      <td>117</td>
    </tr>
    <tr>
      <th>συμπαραμένω</th>
      <td>1</td>
    </tr>
    <tr>
      <th>missing</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ἐξίστημι</th>
      <td>1</td>
    </tr>
    <tr>
      <th>σέβομαι</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ὁράω</th>
      <td>1</td>
    </tr>
    <tr>
      <th>σέβω</th>
      <td>1</td>
    </tr>
    <tr>
      <th>[φοβέω]</th>
      <td>1</td>
    </tr>
    <tr>
      <th>εὐλογέω</th>
      <td>1</td>
    </tr>
    <tr>
      <th>φοβέω|κατανοέω</th>
      <td>1</td>
    </tr>
    <tr>
      <th>φροντίζω</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
plot_data = Greek_verbs_for_yare.sort_values()

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.barh(plot_data.index, plot_data.values, color="#4C78A8")
ax.bar_label(bars, padding=3)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek verb lexeme")
ax.set_title("Greek verbs used to translate Hebrew ירא")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, plot_data.max() * 1.08)
plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_80_0.png)
    


# Greek prepositions in the ירא + מן valence

This section selects the records classified as `MN_complement` and examines whether Hebrew מן is matched by a Greek preposition.


```python
MN_data = df.loc[df["complement type"].eq("MN_complement")].copy()

greek_prep_column = "phrase2word1_prep_GT_lexUTF8"
greek_word3_column = "phrase2word3_prep_GT_lexUTF8"

Greek_prepositions_for_MN = (
    MN_data[greek_prep_column]
    .fillna("[no Greek preposition]")
    .value_counts()
    .rename_axis("Greek preposition")
    .rename("frequency")
)

n_MN = len(MN_data)
n_matched = MN_data[greek_prep_column].notna().sum() - MN_data[greek_prep_column].eq("TC:missing").sum()
n_no_greek_prep = MN_data[greek_prep_column].isna().sum()
n_text_critical = MN_data[greek_prep_column].eq("TC:missing").sum()
n_assessable = n_MN - n_text_critical

MN_match_summary = pd.Series({
    "MN complements": n_MN,
    "matched with a Greek preposition": n_matched,
    "no Greek preposition": n_no_greek_prep,
    "text-critical data missing": n_text_critical,
    "overall match rate (%)": round(n_matched / n_MN * 100, 1),
    "match rate among assessable cases (%)": round(n_matched / n_assessable * 100, 1),
}, name="value")

display(MN_match_summary.to_frame())
display(Greek_prepositions_for_MN.to_frame())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MN complements</th>
      <td>52.0</td>
    </tr>
    <tr>
      <th>matched with a Greek preposition</th>
      <td>32.0</td>
    </tr>
    <tr>
      <th>no Greek preposition</th>
      <td>18.0</td>
    </tr>
    <tr>
      <th>text-critical data missing</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>overall match rate (%)</th>
      <td>61.5</td>
    </tr>
    <tr>
      <th>match rate among assessable cases (%)</th>
      <td>64.0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>Greek preposition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ἀπό</th>
      <td>32</td>
    </tr>
    <tr>
      <th>[no Greek preposition]</th>
      <td>18</td>
    </tr>
    <tr>
      <th>TC:missing</th>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_data = Greek_prepositions_for_MN.sort_values()

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(plot_data.index, plot_data.values, color="#59A14F")
ax.bar_label(bars, padding=3)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek match")
ax.set_title("Greek prepositions matching Hebrew מן")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, plot_data.max() * 1.12)
plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_83_0.png)
    


## Greek lexemes governed by ἀπό

For the MN complements in which Hebrew מן is matched by Greek ἀπό, this subsection shows the distribution recorded in `phrase2word3_prep_GT_lexUTF8`.


```python
MN_with_apo = MN_data.loc[MN_data[greek_prep_column].eq("ἀπό")].copy()

Greek_lexemes_with_apo = (
    MN_with_apo[greek_word3_column]
    .fillna("[no aligned Greek lexeme]")
    .value_counts()
    .rename_axis("Greek lexeme governed by ἀπό")
    .rename("frequency")
)

display(pd.Series({"MN complements with ἀπό": len(MN_with_apo)}, name="value").to_frame())
display(Greek_lexemes_with_apo.to_frame())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>MN complements with ἀπό</th>
      <td>32</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>Greek lexeme governed by ἀπό</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>πρόσωπον</th>
      <td>17</td>
    </tr>
    <tr>
      <th>λόγος</th>
      <td>3</td>
    </tr>
    <tr>
      <th>αὐτός</th>
      <td>2</td>
    </tr>
    <tr>
      <th>ἀκοή</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ἄνθρωπος</th>
      <td>1</td>
    </tr>
    <tr>
      <th>Ἀσσύριος</th>
      <td>1</td>
    </tr>
    <tr>
      <th>κρίμα</th>
      <td>1</td>
    </tr>
    <tr>
      <th>θηρίον</th>
      <td>1</td>
    </tr>
    <tr>
      <th>κακός</th>
      <td>1</td>
    </tr>
    <tr>
      <th>μυριάς</th>
      <td>1</td>
    </tr>
    <tr>
      <th>σύ</th>
      <td>1</td>
    </tr>
    <tr>
      <th>σημεῖον</th>
      <td>1</td>
    </tr>
    <tr>
      <th>φόβος</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_data = Greek_lexemes_with_apo.sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(plot_data.index, plot_data.values, color="#4C78A8")
ax.bar_label(bars, padding=3)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek lexeme governed by ἀπό")
ax.set_title("Greek lexemes occurring with ἀπό in ירא + מן complements")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, plot_data.max() * 1.15)
plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_86_0.png)
    


## MN complements without a Greek preposition

When `phrase2word1_prep_GT_lexUTF8` is empty, the following table and graph show the aligned Greek lexeme recorded in `phrase2word3_prep_GT_lexUTF8`. These are content lexemes, not alternative Greek prepositions.


```python
MN_without_Greek_preposition = MN_data.loc[MN_data[greek_prep_column].isna()].copy()

Greek_lexemes_without_preposition = (
    MN_without_Greek_preposition[greek_word3_column]
    .fillna("[no aligned Greek lexeme]")
    .value_counts()
    .rename_axis("Aligned Greek lexeme")
    .rename("frequency")
)

case_columns = [
    "original_order", "book", "chapter", "verse",
    "phrase2_text_BHS", "phrase2_text_GT", greek_word3_column,
]

display(Greek_lexemes_without_preposition.to_frame())
display(MN_without_Greek_preposition[case_columns].sort_values("original_order"))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>Aligned Greek lexeme</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>θεός</th>
      <td>5</td>
    </tr>
    <tr>
      <th>αὐτός</th>
      <td>4</td>
    </tr>
    <tr>
      <th>κύριος</th>
      <td>2</td>
    </tr>
    <tr>
      <th>πάροδος</th>
      <td>1</td>
    </tr>
    <tr>
      <th>πτόησις</th>
      <td>1</td>
    </tr>
    <tr>
      <th>σύ</th>
      <td>1</td>
    </tr>
    <tr>
      <th>τίς</th>
      <td>1</td>
    </tr>
    <tr>
      <th>λόγος</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ὑμᾶς</th>
      <td>1</td>
    </tr>
    <tr>
      <th>οἶκος</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original_order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>phrase2_text_BHS</th>
      <th>phrase2_text_GT</th>
      <th>phrase2word3_prep_GT_lexUTF8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>105</th>
      <td>3</td>
      <td>Exodus</td>
      <td>9</td>
      <td>30</td>
      <td>מִפְּנֵ֖י יְהוָ֥ה אֱלֹהִֽים׃</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>50</th>
      <td>6</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>14</td>
      <td>מֵּאֱלֹהֶ֖יךָ</td>
      <td>|τὸν|θεόν|σου</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>51</th>
      <td>8</td>
      <td>Leviticus</td>
      <td>19</td>
      <td>32</td>
      <td>מֵּאֱלֹהֶ֖יךָ</td>
      <td>|τὸν|θεόν|σου</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>52</th>
      <td>9</td>
      <td>Leviticus</td>
      <td>25</td>
      <td>17</td>
      <td>מֵֽאֱלֹהֶ֑יךָ</td>
      <td>|τὸν|θεόν|σου</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>53</th>
      <td>10</td>
      <td>Leviticus</td>
      <td>25</td>
      <td>36</td>
      <td>מֵֽאֱלֹהֶ֑יךָ</td>
      <td>|τὸν|θεόν|σου</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>54</th>
      <td>11</td>
      <td>Leviticus</td>
      <td>25</td>
      <td>43</td>
      <td>מֵאֱלֹהֶֽיךָ׃</td>
      <td>|τὸν|θεόν|σου</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>56</th>
      <td>17</td>
      <td>Deuteronomium</td>
      <td>2</td>
      <td>4</td>
      <td>מִכֶּ֔ם</td>
      <td>|ὑμᾶς</td>
      <td>ὑμᾶς</td>
    </tr>
    <tr>
      <th>43</th>
      <td>23</td>
      <td>Deuteronomium</td>
      <td>7</td>
      <td>18</td>
      <td>מֵהֶ֑ם</td>
      <td>|αὐτούς</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>48</th>
      <td>28</td>
      <td>Deuteronomium</td>
      <td>28</td>
      <td>10</td>
      <td>מִמֶּֽךָּ׃</td>
      <td>|σε</td>
      <td>σύ</td>
    </tr>
    <tr>
      <th>44</th>
      <td>34</td>
      <td>Josua</td>
      <td>10</td>
      <td>8</td>
      <td>מֵהֶ֔ם</td>
      <td>|αὐτούς</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>46</th>
      <td>63</td>
      <td>Reges_II</td>
      <td>25</td>
      <td>24</td>
      <td>מֵעַבְדֵ֣י הַכַּשְׂדִּ֑ים</td>
      <td>|πάροδον|τῶν|Χαλδαίων</td>
      <td>πάροδος</td>
    </tr>
    <tr>
      <th>42</th>
      <td>76</td>
      <td>Jeremia</td>
      <td>10</td>
      <td>5</td>
      <td>מֵהֶם֙</td>
      <td>|αὐτά</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>45</th>
      <td>82</td>
      <td>Ezechiel</td>
      <td>2</td>
      <td>6</td>
      <td>מֵהֶ֜ם</td>
      <td>|αὐτούς</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>55</th>
      <td>84</td>
      <td>Ezechiel</td>
      <td>2</td>
      <td>6</td>
      <td>מִדִּבְרֵיהֶ֤ם</td>
      <td>|τοὺς|λόγους|αὐτῶν</td>
      <td>λόγος</td>
    </tr>
    <tr>
      <th>49</th>
      <td>100</td>
      <td>Psalmi</td>
      <td>27</td>
      <td>1</td>
      <td>מִמִּ֣י</td>
      <td>|τίνα</td>
      <td>τίς</td>
    </tr>
    <tr>
      <th>115</th>
      <td>101</td>
      <td>Psalmi</td>
      <td>33</td>
      <td>8</td>
      <td>מֵ֭יְהוָה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>47</th>
      <td>118</td>
      <td>Proverbia</td>
      <td>3</td>
      <td>25</td>
      <td>מִפַּ֣חַד</td>
      <td>|πτόησιν</td>
      <td>πτόησις</td>
    </tr>
    <tr>
      <th>124</th>
      <td>120</td>
      <td>Proverbia</td>
      <td>31</td>
      <td>21</td>
      <td>מִשָּׁ֑לֶג</td>
      <td>τῶν|ἐν|οἴκῳ</td>
      <td>οἶκος</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_data = Greek_lexemes_without_preposition.sort_values()

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(plot_data.index, plot_data.values, color="#F28E2B")
ax.bar_label(bars, padding=3)
ax.set_xlabel("Frequency")
ax.set_ylabel("Aligned Greek lexeme")
ax.set_title("Greek lexemes in MN complements without a Greek preposition")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, plot_data.max() * 1.15)
plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_89_0.png)
    


# Greek realization of direct objects in the ירא + את valence

This section selects records marked `Objc w >T` in `phrase2_function` and examines whether the aligned Greek phrase begins with a preposition.


```python
direct_object_data = df.loc[df["phrase2_function"].eq("Objc w >T")].copy()

greek_prep_column = "phrase2word1_prep_GT_lexUTF8"
greek_word3_column = "phrase2word3_prep_GT_lexUTF8"

Greek_prepositions_for_direct_objects = (
    direct_object_data[greek_prep_column]
    .fillna("[no Greek preposition]")
    .value_counts()
    .rename_axis("Greek phrase-initial preposition")
    .rename("frequency")
)

n_direct_objects = len(direct_object_data)
n_with_preposition = direct_object_data[greek_prep_column].notna().sum()
n_without_preposition = direct_object_data[greek_prep_column].isna().sum()

direct_object_prep_summary = pd.Series({
    "Objc w >T cases": n_direct_objects,
    "with a Greek phrase-initial preposition": n_with_preposition,
    "without a Greek phrase-initial preposition": n_without_preposition,
    "cases with a Greek preposition (%)": round(n_with_preposition / n_direct_objects * 100, 1),
}, name="value")

display(direct_object_prep_summary.to_frame())
display(Greek_prepositions_for_direct_objects.to_frame())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Objc w &gt;T cases</th>
      <td>46.0</td>
    </tr>
    <tr>
      <th>with a Greek phrase-initial preposition</th>
      <td>2.0</td>
    </tr>
    <tr>
      <th>without a Greek phrase-initial preposition</th>
      <td>44.0</td>
    </tr>
    <tr>
      <th>cases with a Greek preposition (%)</th>
      <td>4.3</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>Greek phrase-initial preposition</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>[no Greek preposition]</th>
      <td>44</td>
    </tr>
    <tr>
      <th>ἀπ’</th>
      <td>1</td>
    </tr>
    <tr>
      <th>σὺν</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_data = Greek_prepositions_for_direct_objects.sort_values()

fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.barh(plot_data.index, plot_data.values, color="#59A14F")
ax.bar_label(bars, padding=3)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek phrase-initial preposition")
ax.set_title("Greek prepositions initiating Objc w >T phrases")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, plot_data.max() * 1.12)
plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_92_0.png)
    


## Greek lexemes where no phrase-initial preposition is recorded

For direct-object cases with an empty `phrase2word1_prep_GT_lexUTF8`, the following cells show the distribution in `phrase2word3_prep_GT_lexUTF8` and list the underlying records.


```python
direct_objects_without_Greek_preposition = direct_object_data.loc[
    direct_object_data[greek_prep_column].isna()
].copy()

Greek_lexemes_without_DO_preposition = (
    direct_objects_without_Greek_preposition[greek_word3_column]
    .fillna("[no aligned Greek lexeme]")
    .value_counts()
    .rename_axis("Aligned Greek lexeme")
    .rename("frequency")
)

case_columns = [
    "original_order", "book", "chapter", "verse",
    "phrase2_text_BHS", "phrase2_text_GT", greek_word3_column,
]

display(Greek_lexemes_without_DO_preposition.to_frame())
display(direct_objects_without_Greek_preposition[case_columns].sort_values("original_order"))
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>frequency</th>
    </tr>
    <tr>
      <th>Aligned Greek lexeme</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>κύριος</th>
      <td>19</td>
    </tr>
    <tr>
      <th>θεός</th>
      <td>7</td>
    </tr>
    <tr>
      <th>αὐτός</th>
      <td>6</td>
    </tr>
    <tr>
      <th>ἐγώ</th>
      <td>3</td>
    </tr>
    <tr>
      <th>ὄνομα</th>
      <td>2</td>
    </tr>
    <tr>
      <th>λαός</th>
      <td>2</td>
    </tr>
    <tr>
      <th>Μωσεύς</th>
      <td>1</td>
    </tr>
    <tr>
      <th>βασιλεύς</th>
      <td>1</td>
    </tr>
    <tr>
      <th>φόβος</th>
      <td>1</td>
    </tr>
    <tr>
      <th>οἶκος</th>
      <td>1</td>
    </tr>
    <tr>
      <th>ὅρκος</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original_order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>phrase2_text_BHS</th>
      <th>phrase2_text_GT</th>
      <th>phrase2word3_prep_GT_lexUTF8</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>99</th>
      <td>1</td>
      <td>Exodus</td>
      <td>1</td>
      <td>17</td>
      <td>אֶת־הָ֣אֱלֹהִ֔ים</td>
      <td>τὸν|θεὸν</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>100</th>
      <td>2</td>
      <td>Exodus</td>
      <td>1</td>
      <td>21</td>
      <td>אֶת־הָאֱלֹהִ֑ים</td>
      <td>τὸν|θεὸν</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>106</th>
      <td>4</td>
      <td>Exodus</td>
      <td>14</td>
      <td>31</td>
      <td>אֶת־יְהוָ֑ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>120</th>
      <td>13</td>
      <td>Numeri</td>
      <td>14</td>
      <td>9</td>
      <td>אֶת־עַ֣ם הָאָ֔רֶץ</td>
      <td>τὸν|λαὸν|τῆς|γῆς</td>
      <td>λαός</td>
    </tr>
    <tr>
      <th>57</th>
      <td>15</td>
      <td>Numeri</td>
      <td>21</td>
      <td>34</td>
      <td>אֹתֹ֔ו</td>
      <td>αὐτόν</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>58</th>
      <td>18</td>
      <td>Deuteronomium</td>
      <td>3</td>
      <td>2</td>
      <td>אֹתֹ֔ו</td>
      <td>αὐτόν</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>80</th>
      <td>21</td>
      <td>Deuteronomium</td>
      <td>6</td>
      <td>2</td>
      <td>אֶת־יְהוָ֣ה אֱלֹהֶ֗יךָ</td>
      <td>κύριον|τὸν|θεὸν|ὑμῶν</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>78</th>
      <td>22</td>
      <td>Deuteronomium</td>
      <td>6</td>
      <td>13</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ</td>
      <td>κύριον|τὸν|θεόν|σου</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>79</th>
      <td>24</td>
      <td>Deuteronomium</td>
      <td>10</td>
      <td>20</td>
      <td>אֶת־יְהוָ֧ה אֱלֹהֶ֛יךָ</td>
      <td>κύριον|τὸν|θεόν|σου</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>59</th>
      <td>25</td>
      <td>Deuteronomium</td>
      <td>13</td>
      <td>5</td>
      <td>אֹתֹ֣ו</td>
      <td>αὐτὸν</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>81</th>
      <td>29</td>
      <td>Deuteronomium</td>
      <td>31</td>
      <td>12</td>
      <td>אֶת־יְהוָ֣ה אֱלֹֽהֵיכֶ֔ם</td>
      <td>κύριον|τὸν|θεὸν|ὑμῶν</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>60</th>
      <td>30</td>
      <td>Josua</td>
      <td>4</td>
      <td>14</td>
      <td>אֹתֹ֔ו</td>
      <td>αὐτὸν</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>86</th>
      <td>31</td>
      <td>Josua</td>
      <td>4</td>
      <td>14</td>
      <td>אֶת־מֹשֶׁ֖ה</td>
      <td>μωυσῆν</td>
      <td>Μωσεύς</td>
    </tr>
    <tr>
      <th>82</th>
      <td>32</td>
      <td>Josua</td>
      <td>4</td>
      <td>24</td>
      <td>אֶת־יְהוָ֥ה אֱלֹהֵיכֶ֖ם</td>
      <td>κύριον|τὸν|θεὸν|ὑμῶν</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>76</th>
      <td>36</td>
      <td>Josua</td>
      <td>24</td>
      <td>14</td>
      <td>אֶת־יְהוָ֛ה</td>
      <td>κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>123</th>
      <td>37</td>
      <td>Judices</td>
      <td>6</td>
      <td>10</td>
      <td>אֶת־אֱלֹהֵ֣י הָאֱמֹרִ֔י</td>
      <td>τοὺς|θεοὺς|τοῦ|Αμορραίου</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>121</th>
      <td>38</td>
      <td>Judices</td>
      <td>6</td>
      <td>27</td>
      <td>אֶת־בֵּ֨ית אָבִ֜יו וְאֶת־אַנְשֵׁ֥י הָעִ֛יר</td>
      <td>τὸν|οἶκον|τοῦ|πατρὸς|αὐτοῦ|καὶ|τοὺς|ἄνδρας|τῆς...</td>
      <td>οἶκος</td>
    </tr>
    <tr>
      <th>107</th>
      <td>40</td>
      <td>Samuel_I</td>
      <td>12</td>
      <td>14</td>
      <td>אֶת־יְהוָ֗ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>118</th>
      <td>41</td>
      <td>Samuel_I</td>
      <td>12</td>
      <td>18</td>
      <td>אֶת־יְהוָ֖ה וְאֶת־שְׁמוּאֵֽל׃</td>
      <td>τὸν|κύριον|σφόδρα|καὶ|τὸν|Σαμουηλ</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>108</th>
      <td>42</td>
      <td>Samuel_I</td>
      <td>12</td>
      <td>24</td>
      <td>אֶת־יְהוָ֗ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>122</th>
      <td>43</td>
      <td>Samuel_I</td>
      <td>14</td>
      <td>26</td>
      <td>אֶת־הַשְּׁבֻעָֽה׃</td>
      <td>τὸν|ὅρκον|κυρίου</td>
      <td>ὅρκος</td>
    </tr>
    <tr>
      <th>119</th>
      <td>44</td>
      <td>Samuel_I</td>
      <td>15</td>
      <td>24</td>
      <td>אֶת־הָעָ֔ם</td>
      <td>τὸν|λαὸν</td>
      <td>λαός</td>
    </tr>
    <tr>
      <th>109</th>
      <td>48</td>
      <td>Samuel_II</td>
      <td>6</td>
      <td>9</td>
      <td>אֶת־יְהוָ֖ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>97</th>
      <td>50</td>
      <td>Reges_I</td>
      <td>1</td>
      <td>51</td>
      <td>אֶת־הַמֶּ֣לֶךְ שְׁלֹמֹ֑ה</td>
      <td>τὸν|βασιλέα|Σαλωμων</td>
      <td>βασιλεύς</td>
    </tr>
    <tr>
      <th>110</th>
      <td>55</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>25</td>
      <td>אֶת־יְהוָ֑ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>111</th>
      <td>56</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>28</td>
      <td>אֶת־יְהוָֽה׃</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>61</th>
      <td>58</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>36</td>
      <td>אֹתֹ֣ו</td>
      <td>αὐτὸν</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>117</th>
      <td>61</td>
      <td>Reges_II</td>
      <td>17</td>
      <td>39</td>
      <td>אֶת־יְהוָ֥ה אֱלֹהֵיכֶ֖ם</td>
      <td>τὸν|κύριον|θεὸν|ὑμῶν</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>98</th>
      <td>65</td>
      <td>Jesaia</td>
      <td>8</td>
      <td>12</td>
      <td>אֶת־מֹורָאֹ֥ו</td>
      <td>τὸν|δὲ|φόβον|αὐτοῦ</td>
      <td>φόβος</td>
    </tr>
    <tr>
      <th>65</th>
      <td>71</td>
      <td>Jesaia</td>
      <td>57</td>
      <td>11</td>
      <td>אֹותִ֖י</td>
      <td>ἐμὲ</td>
      <td>ἐγώ</td>
    </tr>
    <tr>
      <th>95</th>
      <td>72</td>
      <td>Jesaia</td>
      <td>59</td>
      <td>19</td>
      <td>אֶת־שֵׁ֣ם יְהוָ֔ה</td>
      <td>τὸ|ὄνομα|κυρίου</td>
      <td>ὄνομα</td>
    </tr>
    <tr>
      <th>66</th>
      <td>74</td>
      <td>Jeremia</td>
      <td>5</td>
      <td>22</td>
      <td>אֹותִ֨י</td>
      <td>ἐμὲ</td>
      <td>ἐγώ</td>
    </tr>
    <tr>
      <th>77</th>
      <td>75</td>
      <td>Jeremia</td>
      <td>5</td>
      <td>24</td>
      <td>אֶת־יְהוָ֣ה אֱלֹהֵ֔ינוּ</td>
      <td>κύριον|τὸν|θεὸν|ἡμῶν</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>112</th>
      <td>78</td>
      <td>Jeremia</td>
      <td>26</td>
      <td>19</td>
      <td>אֶת־יְהוָה֒</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>113</th>
      <td>87</td>
      <td>Hosea</td>
      <td>10</td>
      <td>3</td>
      <td>אֶת־יְהוָ֔ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>114</th>
      <td>89</td>
      <td>Jona</td>
      <td>1</td>
      <td>16</td>
      <td>אֶת־יְהוָ֑ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>83</th>
      <td>93</td>
      <td>Zephania</td>
      <td>3</td>
      <td>7</td>
      <td>אֹותִי֙</td>
      <td>με</td>
      <td>ἐγώ</td>
    </tr>
    <tr>
      <th>116</th>
      <td>102</td>
      <td>Psalmi</td>
      <td>34</td>
      <td>10</td>
      <td>אֶת־יְהוָ֣ה</td>
      <td>τὸν|κύριον</td>
      <td>κύριος</td>
    </tr>
    <tr>
      <th>62</th>
      <td>105</td>
      <td>Psalmi</td>
      <td>67</td>
      <td>8</td>
      <td>אֹ֝תֹ֗ו</td>
      <td>αὐτὸν</td>
      <td>αὐτός</td>
    </tr>
    <tr>
      <th>96</th>
      <td>108</td>
      <td>Psalmi</td>
      <td>102</td>
      <td>16</td>
      <td>אֶת־שֵׁ֣ם יְהוָ֑ה</td>
      <td>τὸ|ὄνομα|κυρίου</td>
      <td>ὄνομα</td>
    </tr>
    <tr>
      <th>101</th>
      <td>117</td>
      <td>Proverbia</td>
      <td>3</td>
      <td>7</td>
      <td>אֶת־יְ֝הוָ֗ה</td>
      <td>τὸν|θεὸν</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>102</th>
      <td>119</td>
      <td>Proverbia</td>
      <td>24</td>
      <td>21</td>
      <td>אֶת־יְהוָ֣ה</td>
      <td>τὸν|θεὸν</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>103</th>
      <td>124</td>
      <td>Ecclesiastes</td>
      <td>12</td>
      <td>13</td>
      <td>אֶת־הָאֱלֹהִ֤ים</td>
      <td>Τὸν|θεὸν</td>
      <td>θεός</td>
    </tr>
    <tr>
      <th>104</th>
      <td>126</td>
      <td>Chronica_I</td>
      <td>13</td>
      <td>12</td>
      <td>אֶת־הָ֣אֱלֹהִ֔ים</td>
      <td>τὸν|θεὸν</td>
      <td>θεός</td>
    </tr>
  </tbody>
</table>
</div>



```python
plot_data = Greek_lexemes_without_DO_preposition.sort_values()

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.barh(plot_data.index, plot_data.values, color="#F28E2B")
ax.bar_label(bars, padding=3)
ax.set_xlabel("Frequency")
ax.set_ylabel("Aligned Greek lexeme")
ax.set_title("Greek lexemes in Objc w >T phrases without a Greek preposition")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlim(0, plot_data.max() * 1.12)
plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_95_0.png)
    



```python

```


```python

```

# Comparison of Hebrew and Greek negation

This section compares the functional negation categories: Hebrew `AL` with Greek `μή`/`μηδέ`, Hebrew `LO` with Greek `οὐ`/`οὐκ`/`οὐ μή`, and absence of negation with absence of negation. Text-critical missing data are reported separately.


```python
negation_data = df.copy()

negation_equivalents = {
    "AL": {"μή", "μηδέ"},
    "LO": {"οὐ", "οὐκ", "οὐ μή"},
}

def compare_negations(row):
    bhs_negation = row["BHS_negation"]
    gt_negation = row["GT_negation"]
    if gt_negation == "TC:missing":
        return "not assessable: TC missing"
    if pd.isna(bhs_negation) and pd.isna(gt_negation):
        return "match"
    if bhs_negation in negation_equivalents and gt_negation in negation_equivalents[bhs_negation]:
        return "match"
    return "mismatch"

negation_data["negation_comparison"] = negation_data.apply(compare_negations, axis=1)

negation_crosstab = pd.crosstab(
    negation_data["BHS_negation"].fillna("[no negation]"),
    negation_data["GT_negation"].fillna("[no negation]"),
)
comparison_counts = negation_data["negation_comparison"].value_counts()
n_assessable = comparison_counts.get("match", 0) + comparison_counts.get("mismatch", 0)
negated_assessable = negation_data.loc[
    negation_data["BHS_negation"].notna()
    & negation_data["negation_comparison"].ne("not assessable: TC missing")
]

negation_match_summary = pd.Series({
    "all records": len(negation_data),
    "assessable records": n_assessable,
    "matches": comparison_counts.get("match", 0),
    "mismatches": comparison_counts.get("mismatch", 0),
    "text-critical/unassessable": comparison_counts.get("not assessable: TC missing", 0),
    "overall match rate among assessable cases (%)": round(comparison_counts.get("match", 0) / n_assessable * 100, 1),
    "match rate among explicitly negated assessable cases (%)": round((negated_assessable["negation_comparison"] == "match").mean() * 100, 1),
}, name="value")

display(negation_match_summary.to_frame())
display(negation_crosstab)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[4], line 1
    ----> 1 negation_data = df.copy()
          3 negation_equivalents = {
          4     "AL": {"μή", "μηδέ"},
          5     "LO": {"οὐ", "οὐκ", "οὐ μή"},
          6 }
          8 def compare_negations(row):


    NameError: name 'df' is not defined



```python
mismatch_columns = [
    "original_order", "book", "chapter", "verse",
    "clause_text_BHS", "clause_text_GT", "BHS_negation", "GT_negation",
]

negation_mismatches = negation_data.loc[
    negation_data["negation_comparison"] == "mismatch",
    mismatch_columns,
].sort_values("original_order")

negation_mismatches
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>original_order</th>
      <th>book</th>
      <th>chapter</th>
      <th>verse</th>
      <th>clause_text_BHS</th>
      <th>clause_text_GT</th>
      <th>BHS_negation</th>
      <th>GT_negation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22</th>
      <td>16</td>
      <td>Deuteronomium</td>
      <td>1</td>
      <td>29</td>
      <td>וְֽלֹא־תִֽירְא֖וּן מֵהֶֽם׃</td>
      <td>καὶ εἶπα πρὸς ὑμᾶς μὴ πτήξητε μηδὲ φοβηθῆτε ἀπ...</td>
      <td>LO</td>
      <td>μηδέ</td>
    </tr>
    <tr>
      <th>33</th>
      <td>85</td>
      <td>Ezechiel</td>
      <td>3</td>
      <td>9</td>
      <td>לֹֽא־תִירָ֤א אֹותָם֙</td>
      <td>καὶ ἔσται διὰ παντὸς κραταιότερον πέτρας μὴ φο...</td>
      <td>LO</td>
      <td>μή</td>
    </tr>
    <tr>
      <th>46</th>
      <td>97</td>
      <td>Maleachi</td>
      <td>3</td>
      <td>5</td>
      <td>וְלֹ֣א יְרֵא֔וּנִי</td>
      <td>καὶ προσάξω πρὸς ὑμᾶς ἐν κρίσει καὶ ἔσομαι μάρ...</td>
      <td>LO</td>
      <td>μή</td>
    </tr>
    <tr>
      <th>4</th>
      <td>114</td>
      <td>Iob</td>
      <td>5</td>
      <td>22</td>
      <td>וּֽמֵחַיַּ֥ת הָ֝אָ֗רֶץ אַל־תִּירָֽא׃</td>
      <td>ἀδίκων καὶ ἀνόμων καταγελάσῃ ἀπὸ δὲ θηρίων ἀγρ...</td>
      <td>AL</td>
      <td>οὐ μή</td>
    </tr>
    <tr>
      <th>16</th>
      <td>118</td>
      <td>Proverbia</td>
      <td>3</td>
      <td>25</td>
      <td>אַל־תִּ֭ירָא מִפַּ֣חַד פִּתְאֹ֑ם</td>
      <td>καὶ οὐ φοβηθήσῃ πτόησιν ἐπελθοῦσαν οὐδὲ ὁρμὰς ...</td>
      <td>AL</td>
      <td>οὐ</td>
    </tr>
  </tbody>
</table>
</div>



## Greek phrase-initial prepositions by Greek negation type

The following table and graphs show how `phrase2word1_prep_GT_lexUTF8` is distributed across the assessable Greek negation types. Empty values are retained as `[no Greek preposition]`; `absent` and `TC:missing` remain separate data categories.


```python
valid_GT_negations = ["μή", "μηδέ", "οὐ", "οὐκ", "οὐ μή"]
negated_GT_data = negation_data.loc[
    negation_data["GT_negation"].isin(valid_GT_negations)
].copy()
negated_GT_data["Greek phrase-initial preposition"] = (
    negated_GT_data["phrase2word1_prep_GT_lexUTF8"]
    .fillna("[no Greek preposition]")
)

prepositions_by_GT_negation = pd.crosstab(
    negated_GT_data["GT_negation"],
    negated_GT_data["Greek phrase-initial preposition"],
).reindex(valid_GT_negations, fill_value=0)
prepositions_by_GT_negation_pct = (
    prepositions_by_GT_negation
    .div(prepositions_by_GT_negation.sum(axis=1), axis=0)
    .mul(100)
    .round(1)
)

display(prepositions_by_GT_negation)
display(prepositions_by_GT_negation_pct)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Greek phrase-initial preposition</th>
      <th>TC:missing</th>
      <th>[no Greek preposition]</th>
      <th>absent</th>
      <th>ἀπό</th>
      <th>ἀπ’</th>
    </tr>
    <tr>
      <th>GT_negation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μή</th>
      <td>1</td>
      <td>11</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <th>μηδέ</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>οὐ</th>
      <td>0</td>
      <td>9</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
    </tr>
    <tr>
      <th>οὐκ</th>
      <td>0</td>
      <td>6</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>οὐ μή</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th>Greek phrase-initial preposition</th>
      <th>TC:missing</th>
      <th>[no Greek preposition]</th>
      <th>absent</th>
      <th>ἀπό</th>
      <th>ἀπ’</th>
    </tr>
    <tr>
      <th>GT_negation</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>μή</th>
      <td>4.8</td>
      <td>52.4</td>
      <td>0.0</td>
      <td>38.1</td>
      <td>4.8</td>
    </tr>
    <tr>
      <th>μηδέ</th>
      <td>50.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>οὐ</th>
      <td>0.0</td>
      <td>64.3</td>
      <td>7.1</td>
      <td>28.6</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>οὐκ</th>
      <td>0.0</td>
      <td>100.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>οὐ μή</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>33.3</td>
      <td>66.7</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>



```python
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

prepositions_by_GT_negation.plot(kind="bar", stacked=True, ax=axes[0])
axes[0].set_title("Greek phrase-initial prepositions by negation type")
axes[0].set_xlabel("Greek negation")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis="x", rotation=0)
axes[0].legend(title="Greek preposition", fontsize=8)

prepositions_by_GT_negation_pct.plot(kind="bar", stacked=True, ax=axes[1])
axes[1].set_title("Within-negation distribution")
axes[1].set_xlabel("Greek negation")
axes[1].set_ylabel("Percentage")
axes[1].set_ylim(0, 100)
axes[1].tick_params(axis="x", rotation=0)
axes[1].legend(title="Greek preposition", fontsize=8)

for ax in axes:
    ax.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
plt.show()
```


    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_103_0.png)
    



```python

```

## General Comparative Distribution of the Hebrew and Greek "to fear"

The following cell counts the Greek predicate values in **word G lex Pred** that are aligned with the Hebrew predicate ירא in **word H lexUTF8 Pred**. The field is retained as recorded in the research workbook, so multi-word or contextually expanded predicate values remain visible. Missing Greek predicate values are reported explicitly as **[no aligned Greek predicate]**.


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Source workbook and output location (relative to the notebook directory).
data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "001_GeneralFearDistribution.png"

fear_data = pd.read_excel(data_path, sheet_name="Sheet1")
fear_data = fear_data.loc[
    fear_data["word H lexUTF8 Pred"].eq("ירא")
    & fear_data["HIPHIL_usable"].eq("y"),
    ["word H lexUTF8 Pred", "word G lex Pred", "HIPHIL_usable"],
].copy()

# Keep missing alignments in the distribution so that the totals reconcile to
# the complete Hebrew ירא dataset.
fear_data["Greek predicate"] = fear_data["word G lex Pred"].fillna(
    "[no aligned Greek predicate]"
)
greek_verb_distribution = (
    fear_data["Greek predicate"]
    .value_counts()
    .rename_axis("Greek verb / predicate value")
    .reset_index(name="Frequency")
)
greek_verb_distribution["Percentage"] = (
    greek_verb_distribution["Frequency"]
    .div(greek_verb_distribution["Frequency"].sum())
    .mul(100)
    .round(1)
)

print(
    f"Hebrew predicate: ירא | HIPHIL_usable = y | aligned cases: {len(fear_data)}"
)
display(greek_verb_distribution)

# Horizontal bars keep the long recorded predicate values legible.
plot_data = greek_verb_distribution.sort_values("Frequency", ascending=True)
fig_height = max(4.8, 0.42 * len(plot_data) + 1.5)
fig, ax = plt.subplots(figsize=(10, fig_height))
bars = ax.barh(
    plot_data["Greek verb / predicate value"],
    plot_data["Frequency"],
    color="#2E74B5",
    edgecolor="white",
    linewidth=0.7,
)
ax.set_title("Greek predicate values aligned with Hebrew ארי", pad=12)
ax.set_xlabel("Number of aligned cases")
ax.set_ylabel("Greek verb / predicate value")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.bar_label(bars, labels=[str(v) for v in plot_data["Frequency"]], padding=4)
ax.set_xlim(0, plot_data["Frequency"].max() * 1.12)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")
```

    Hebrew predicate: ירא | HIPHIL_usable = y | aligned cases: 115



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Greek verb / predicate value</th>
      <th>Frequency</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>φοβέω</td>
      <td>112</td>
      <td>97.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>σέβομαι</td>
      <td>2</td>
      <td>1.7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>εὐλογέω</td>
      <td>1</td>
      <td>0.9</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_106_2.png)
    


    Saved graphic to: article_images/001_GeneralFearDistribution.png



```python

```

## Having Respect and its Greek Rendering

This Hebrew-only distribution uses the curated rows marked **usable = y**. Here **MN_complement** represents ירא + מִן (“to be afraid of”), while **DO_complement** represents ירא + a direct object (“to respect/revere”).


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "002_Hebrew_Fear_vs_Respect_valence.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
usable_hebrew = df.loc[df["usable"].eq("y")].copy()
complement_order = ["MN_complement", "DO_complement"]
complement_distribution = (usable_hebrew["complement type"].value_counts().reindex(complement_order, fill_value=0).rename_axis("Hebrew complement type").reset_index(name="Frequency"))
complement_distribution["Display label"] = complement_distribution["Hebrew complement type"].map({
    "MN_complement": "ארי + ןמ",
    "DO_complement": "ארי + direct object",
})
complement_distribution["Percentage"] = complement_distribution["Frequency"].div(complement_distribution["Frequency"].sum()).mul(100).round(1)
print(f"Usable Hebrew cases (usable = y): {len(usable_hebrew)}")
display(complement_distribution)

fig, ax = plt.subplots(figsize=(8, 5.2))
bars = ax.bar(complement_distribution["Display label"], complement_distribution["Percentage"], color=["#B45F06", "#2E74B5"], width=0.58)
ax.set_title("Hebrew ארי valence distribution", pad=12)
ax.set_xlabel("Hebrew complement type")
ax.set_ylabel("Percentage of usable cases")
ax.set_ylim(0, 70)
ax.grid(axis="y", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
labels = [f"{p:.1f}%\n(n={n})" for p, n in zip(complement_distribution["Percentage"], complement_distribution["Frequency"])]
ax.bar_label(bars, labels=labels, padding=4)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")
```

    Usable Hebrew cases (usable = y): 124



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hebrew complement type</th>
      <th>Frequency</th>
      <th>Display label</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MN_complement</td>
      <td>52</td>
      <td>ארי + ןמ</td>
      <td>41.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>DO_complement</td>
      <td>72</td>
      <td>ארי + direct object</td>
      <td>58.1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_109_2.png)
    


    Saved graphic to: article_images/002_Hebrew_Fear_vs_Respect_valence.png


## Greek renderings of the Hebrew direct-object construction

This analysis isolates the dominant Greek predicate, φοβέω, in the Hebrew **DO_complement** cases marked **HIPHIL_usable = y**. It lists the aligned values in **word G text Cmpl-element-2** and **word G lex Cmpl-element-2** in descending frequency order.


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "003_a_Hebrew_DO-constructions_in_Greek.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
do_greek = df.loc[
    df["HIPHIL_usable"].eq("y")
    & df["complement type"].eq("DO_complement")
    & df["word G lex Pred"].eq("φοβέω")
].copy()

def frequency_table(column):
    return (
        do_greek[column]
        .fillna("[missing]")
        .value_counts()
        .rename_axis(column)
        .reset_index(name="Frequency")
    )

text_cmpl2_distribution = frequency_table("word G text Cmpl-element-2")
lex_cmpl2_distribution = frequency_table("word G lex Cmpl-element-2")

print(f"Filtered DO_complement cases: {len(do_greek)}")
display(text_cmpl2_distribution)
display(lex_cmpl2_distribution)

fig, axes = plt.subplots(1, 2, figsize=(16, 10))
for ax, table, column, title in [
    (axes[0], text_cmpl2_distribution, "word G text Cmpl-element-2", "Greek complement text"),
    (axes[1], lex_cmpl2_distribution, "word G lex Cmpl-element-2", "Greek complement lexemes"),
]:
    plot_table = table.sort_values("Frequency", ascending=True)
    bars = ax.barh(plot_table[column].astype(str), plot_table["Frequency"], color="#2E74B5")
    ax.set_title(title, pad=10)
    ax.set_xlabel("Frequency")
    ax.set_ylabel("")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.bar_label(bars, labels=[str(v) for v in plot_table["Frequency"]], padding=3)
    ax.set_xlim(0, plot_table["Frequency"].max() * 1.18)
fig.suptitle("Greek renderings of Hebrew ארי + direct object with φοβέω", y=1.01, fontsize=16)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")
```

    Filtered DO_complement cases: 65



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word G text Cmpl-element-2</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>κύριον</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>αὐτὸν</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>θεόν</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>θεοὺς</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>με</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>αὐτόν</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>σε</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ἁγίων μου</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>θεον</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>θεὸν</td>
      <td>2</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ὄνομα</td>
      <td>2</td>
    </tr>
    <tr>
      <th>11</th>
      <td>λαὸν</td>
      <td>2</td>
    </tr>
    <tr>
      <th>12</th>
      <td>φόβον</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>φόβῳ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>φόβον αὐτοῦ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ἐμὲ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>τίνα</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>οἶκον</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ὀνειδισμὸν</td>
      <td>1</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ῥομφαίαν</td>
      <td>1</td>
    </tr>
    <tr>
      <th>20</th>
      <td>πατέρα αὐτοῦ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>21</th>
      <td>ὅρκον</td>
      <td>1</td>
    </tr>
    <tr>
      <th>22</th>
      <td>μωυσῆν</td>
      <td>1</td>
    </tr>
    <tr>
      <th>23</th>
      <td>κακά</td>
      <td>1</td>
    </tr>
    <tr>
      <th>24</th>
      <td>βασιλέα</td>
      <td>1</td>
    </tr>
    <tr>
      <th>25</th>
      <td>αὐτῶν</td>
      <td>1</td>
    </tr>
    <tr>
      <th>26</th>
      <td>αὐτούς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>27</th>
      <td>ὁδούς σου</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word G lex Cmpl-element-2</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>κύριος</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>θεός</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>αὐτός</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ἐγώ</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>φόβος</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ὄνομα</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>σύ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>λαός</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ἅγιος</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>τίς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>οἶκος</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ὅρκος</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ῥομφαία</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>πατήρ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>βασιλεύς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ὀνειδισμός</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Μωσεύς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>κακός</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ὁδός</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_111_3.png)
    


    Saved graphic to: article_images/003_a_Hebrew_DO-constructions_in_Greek.png



```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "003_b_Hebrew_DO-constructions_in_Greek.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
do_greek = df.loc[
    df["HIPHIL_usable"].eq("y")
    & df["complement type"].eq("DO_complement")
    & df["word G lex Pred"].eq("φοβέω")
].copy()

lex_cmpl2_distribution = (
    do_greek["word G lex Cmpl-element-2"]
    .fillna("[missing]")
    .value_counts()
    .rename_axis("word G lex Cmpl-element-2")
    .reset_index(name="Frequency")
)

print(f"Filtered Jara + direct object cases with φοβέω: {len(do_greek)}")
display(lex_cmpl2_distribution)

plot_data = lex_cmpl2_distribution.sort_values("Frequency", ascending=True)
fig_height = max(7, 0.42 * len(plot_data) + 1.8)
fig, ax = plt.subplots(figsize=(10, fig_height))
bars = ax.barh(
    plot_data["word G lex Cmpl-element-2"].astype(str),
    plot_data["Frequency"],
    color="#2E74B5",
)
ax.set_title(
    "Greek complement lexemes in Hebrew ארי + direct object with φοβέω",
    pad=12,
)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek complement lexeme (word G lex Cmpl-element-2)")
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.bar_label(
    bars,
    labels=[str(value) for value in plot_data["Frequency"]],
    padding=3,
)
ax.set_xlim(0, plot_data["Frequency"].max() * 1.18)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")

```

    Filtered Jara + direct object cases with φοβέω: 65



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word G lex Cmpl-element-2</th>
      <th>Frequency</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>κύριος</td>
      <td>17</td>
    </tr>
    <tr>
      <th>1</th>
      <td>θεός</td>
      <td>14</td>
    </tr>
    <tr>
      <th>2</th>
      <td>αὐτός</td>
      <td>9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>ἐγώ</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>φόβος</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ὄνομα</td>
      <td>2</td>
    </tr>
    <tr>
      <th>6</th>
      <td>σύ</td>
      <td>2</td>
    </tr>
    <tr>
      <th>7</th>
      <td>λαός</td>
      <td>2</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ἅγιος</td>
      <td>2</td>
    </tr>
    <tr>
      <th>9</th>
      <td>τίς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>οἶκος</td>
      <td>1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ὅρκος</td>
      <td>1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ῥομφαία</td>
      <td>1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>πατήρ</td>
      <td>1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>βασιλεύς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ὀνειδισμός</td>
      <td>1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Μωσεύς</td>
      <td>1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>κακός</td>
      <td>1</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ὁδός</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_112_2.png)
    


    Saved graphic to: article_images/003_b_Hebrew_DO-constructions_in_Greek.png


## Being Afraid (ירא + מִן) and Its Greek Rendering

This analysis isolates the Hebrew **MN_complement** cases marked **HIPHIL_usable = y**. The Greek rendering is measured through the aligned values in **word G lex Cmpl-element-1**, allowing the frequency of ἀπό and the non-prepositional renderings to be compared directly.


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Reload the latest workbook before calculating the distribution.
data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "004_a_Hebrew_Fear_Valence_in_Greek.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
mn_greek = df.loc[
    df["HIPHIL_usable"].eq("y")
    & df["complement type"].eq("MN_complement")
].copy()

mn_distribution = (
    mn_greek["word G lex Cmpl-element-1"]
    .fillna("[missing]")
    .value_counts()
    .rename_axis("Greek rendering of Hebrew MN complement")
    .reset_index(name="Frequency")
)
mn_distribution["Percentage"] = (
    mn_distribution["Frequency"]
    .div(mn_distribution["Frequency"].sum())
    .mul(100)
    .round(1)
)

print(f"Filtered MN_complement cases: {len(mn_greek)}")
display(mn_distribution)

plot_data = mn_distribution.sort_values("Frequency", ascending=True)
fig, ax = plt.subplots(figsize=(9, 6.5))
bars = ax.barh(
    plot_data["Greek rendering of Hebrew MN complement"].astype(str),
    plot_data["Percentage"],
    color="#B45F06",
)
ax.set_title("Greek renderings of Hebrew ארי + ןמ", pad=12)
ax.set_xlabel("Percentage of usable MN_complement cases")
ax.set_ylabel("Greek rendering (word G lex Cmpl-element-1)")
ax.set_xlim(0, plot_data["Percentage"].max() * 1.18)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.bar_label(
    bars,
    labels=[f"{p:.1f}% (n={n})" for p, n in zip(plot_data["Percentage"], plot_data["Frequency"])],
    padding=4,
)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")
```

    Filtered MN_complement cases: 47



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Greek rendering of Hebrew MN complement</th>
      <th>Frequency</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>ἀπό</td>
      <td>32</td>
      <td>68.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>θεός</td>
      <td>5</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>2</th>
      <td>αὐτός</td>
      <td>4</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>κύριος</td>
      <td>2</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>πτόησις</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>σύ</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>τίς</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ὑμεῖς</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_114_2.png)
    


    Saved graphic to: article_images/004_a_Hebrew_Fear_Valence_in_Greek.png



```python
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "004_b_Hebrew_Fear_Valence_in_Greek.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
mn_greek = df.loc[
    df["HIPHIL_usable"].eq("y")
    & df["complement type"].eq("MN_complement")
].copy()

# Compare each element-2 value with whether it is preceded by ἀπό. Every
# combination becomes one aligned row across the two graph sections.
mn_greek["Element-1 status"] = np.where(
    mn_greek["word G lex Cmpl-element-1"].eq("ἀπό"),
    "ἀπό",
    "absent",
)
mn_greek["Element-2 value"] = mn_greek[
    "word G lex Cmpl-element-2"
].fillna("[missing]")

combo_distribution = (
    mn_greek.groupby(
        ["Element-1 status", "Element-2 value"],
        dropna=False,
    )
    .size()
    .reset_index(name="Frequency")
)
combo_distribution["Percentage"] = (
    combo_distribution["Frequency"]
    .div(len(mn_greek))
    .mul(100)
    .round(1)
)
combo_distribution["status_order"] = combo_distribution[
    "Element-1 status"
].map({"ἀπό": 0, "absent": 1})
combo_distribution = (
    combo_distribution.sort_values(
        ["status_order", "Frequency", "Element-2 value"],
        ascending=[False, True, True],
    )
    .drop(columns="status_order")
    .reset_index(drop=True)
)

print(f"Usable MN_complement cases: {len(mn_greek)}")
display(combo_distribution)

y = np.arange(len(combo_distribution))
fig_height = max(8, 0.52 * len(combo_distribution) + 2)
fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, fig_height),
    sharey=False,
    gridspec_kw={"width_ratios": [1, 1.35], "wspace": 0.18},
)

status_colors = combo_distribution["Element-1 status"].map(
    {"ἀπό": "#B45F06", "absent": "#8C8C8C"}
)
left_bars = axes[0].barh(
    y,
    combo_distribution["Frequency"],
    color=status_colors,
)
axes[0].set_yticks(y)
axes[0].set_yticklabels(combo_distribution["Element-1 status"])
axes[0].set_title("Preceding Greek ἀπό or absent", pad=10)
axes[0].set_xlabel("Combination frequency")
axes[0].set_ylabel("word G lex Cmpl-element-1")

right_bars = axes[1].barh(
    y,
    combo_distribution["Frequency"],
    color="#2E74B5",
)
axes[1].set_yticks(y)
axes[1].set_yticklabels(combo_distribution["Element-2 value"])
axes[1].set_ylim(axes[0].get_ylim())
axes[1].set_title("Following Greek lexical value", pad=10)
axes[1].set_xlabel("Combination frequency")
axes[1].set_ylabel("word G lex Cmpl-element-2")

bar_labels = [
    f"{percentage:.1f}% (n={frequency})"
    for percentage, frequency in zip(
        combo_distribution["Percentage"],
        combo_distribution["Frequency"],
    )
]
for ax, bars in zip(axes, [left_bars, right_bars]):
    ax.bar_label(bars, labels=bar_labels, padding=4)
    ax.set_xlim(0, combo_distribution["Frequency"].max() * 1.42)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle=":", alpha=0.45)
    ax.set_axisbelow(True)

fig.suptitle(
    "Greek realizations of Hebrew Jara + Min: preceding ἀπό and following value",
    y=1.01,
    fontsize=16,
)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")

```

    Usable MN_complement cases: 47



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Element-1 status</th>
      <th>Element-2 value</th>
      <th>Frequency</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>absent</td>
      <td>πτόησις</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>absent</td>
      <td>σύ</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>absent</td>
      <td>τίς</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>absent</td>
      <td>ὑμεῖς</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>absent</td>
      <td>κύριος</td>
      <td>2</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>absent</td>
      <td>αὐτός</td>
      <td>4</td>
      <td>8.5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>absent</td>
      <td>θεός</td>
      <td>5</td>
      <td>10.6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>ἀπό</td>
      <td>θηρίον</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>ἀπό</td>
      <td>κακός</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ἀπό</td>
      <td>κρίμα</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>ἀπό</td>
      <td>μυριάς</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>ἀπό</td>
      <td>σημεῖον</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>ἀπό</td>
      <td>σύ</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>ἀπό</td>
      <td>φόβος</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>ἀπό</td>
      <td>ἀκοή</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>15</th>
      <td>ἀπό</td>
      <td>ἄνθρωπος</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>16</th>
      <td>ἀπό</td>
      <td>Ἀσσύριος</td>
      <td>1</td>
      <td>2.1</td>
    </tr>
    <tr>
      <th>17</th>
      <td>ἀπό</td>
      <td>αὐτός</td>
      <td>2</td>
      <td>4.3</td>
    </tr>
    <tr>
      <th>18</th>
      <td>ἀπό</td>
      <td>λόγος</td>
      <td>3</td>
      <td>6.4</td>
    </tr>
    <tr>
      <th>19</th>
      <td>ἀπό</td>
      <td>πρόσωπον</td>
      <td>17</td>
      <td>36.2</td>
    </tr>
  </tbody>
</table>
</div>


    /var/folders/y4/np6563rn613c38dx_w2wwcxm0000gp/T/ipykernel_11606/2258294740.py:113: UserWarning: This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.
      fig.tight_layout()



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_115_3.png)
    


    Saved graphic to: article_images/004_b_Hebrew_Fear_Valence_in_Greek.png


## Greek nouns governed by φοβέω + ἀπό

This analysis retains the Hebrew MN-complement filter and narrows the Greek rendering to cases in which **word G lex Cmpl-element-1** is ἀπό. The graph shows the distribution of the aligned governed nouns recorded in **word G lex Cmpl-element-2**.


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Reload the latest workbook before calculating the distribution.
data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "005_Greek_Fear_Valence_goverened_nouns.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
apo_mn = df.loc[
    df["HIPHIL_usable"].eq("y")
    & df["complement type"].eq("MN_complement")
    & df["word G lex Cmpl-element-1"].eq("ἀπό")
].copy()

governed_noun_distribution = (
    apo_mn["word G lex Cmpl-element-2"]
    .fillna("[missing]")
    .value_counts()
    .rename_axis("Greek governed noun")
    .reset_index(name="Frequency")
)
governed_noun_distribution["Percentage"] = (
    governed_noun_distribution["Frequency"]
    .div(governed_noun_distribution["Frequency"].sum())
    .mul(100)
    .round(1)
)

print(f"Filtered MN_complement cases with ἀπό: {len(apo_mn)}")
display(governed_noun_distribution)

plot_data = governed_noun_distribution.sort_values("Frequency", ascending=True)
fig, ax = plt.subplots(figsize=(9, 9))
bars = ax.barh(plot_data["Greek governed noun"].astype(str), plot_data["Frequency"], color="#2E74B5")
ax.set_title("Greek nouns governed by φοβέω + ἀπό", pad=12)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek governed noun (word G lex Cmpl-element-2)")
ax.set_xlim(0, plot_data["Frequency"].max() * 1.22)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.bar_label(
    bars,
    labels=[f"{p:.1f}% (n={n})" for p, n in zip(plot_data["Percentage"], plot_data["Frequency"])],
    padding=4,
)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")
```

    Filtered MN_complement cases with ἀπό: 32



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Greek governed noun</th>
      <th>Frequency</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>πρόσωπον</td>
      <td>14</td>
      <td>43.8</td>
    </tr>
    <tr>
      <th>1</th>
      <td>ἀπο</td>
      <td>3</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-</td>
      <td>2</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>αὐτός</td>
      <td>2</td>
      <td>6.2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>λόγος</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>κρίμα</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>ἄνθρωπος</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>κακός</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>θηρίον</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>ἀκοή</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>10</th>
      <td>φόβος</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>11</th>
      <td>μυριάς</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>12</th>
      <td>σύ</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>13</th>
      <td>σημεῖον</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Ἀσσύριος</td>
      <td>1</td>
      <td>3.1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_117_2.png)
    


    Saved graphic to: article_images/005_Greek_Fear_Valence_goverened_nouns.png


## Greek nouns governed by φοβέω + ἀπό + πρόσωπον

This analysis retains the preceding filters and adds **word G lex Cmpl-element-2 = πρόσωπον**. It then shows the distribution of the next aligned Greek lexical value in **word G lex Cmpl-element-3**.


```python
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

# Reload the latest workbook before calculating the distribution.
data_path = Path("data_export/FearValResearchData_combined_woArt_FINAL.xlsx")
image_dir = Path("article_images")
image_dir.mkdir(parents=True, exist_ok=True)
image_path = image_dir / "006_Greek_Fear_Valence_goverened_nouns_withProsopon.png"

df = pd.read_excel(data_path, sheet_name="Sheet1")
prosopon_cases = df.loc[
    df["HIPHIL_usable"].eq("y")
    & df["complement type"].eq("MN_complement")
    & df["word G lex Cmpl-element-1"].eq("ἀπό")
    & df["word G lex Cmpl-element-2"].eq("πρόσωπον")
].copy()

prosopon_distribution = (
    prosopon_cases["word G lex Cmpl-element-3"]
    .fillna("[missing]")
    .value_counts()
    .rename_axis("Greek lexical value after πρόσωπον")
    .reset_index(name="Frequency")
)
prosopon_distribution["Percentage"] = (
    prosopon_distribution["Frequency"]
    .div(prosopon_distribution["Frequency"].sum())
    .mul(100)
    .round(1)
)

print(f"Filtered cases with πρόσωπον: {len(prosopon_cases)}")
display(prosopon_distribution)

plot_data = prosopon_distribution.sort_values("Frequency", ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
bars = ax.barh(plot_data["Greek lexical value after πρόσωπον"].astype(str), plot_data["Frequency"], color="#2E74B5")
ax.set_title("Greek values following φοβέω + ἀπό + πρόσωπον", pad=12)
ax.set_xlabel("Frequency")
ax.set_ylabel("Greek lexical value (word G lex Cmpl-element-3)")
ax.set_xlim(0, plot_data["Frequency"].max() * 1.25)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="x", linestyle=":", alpha=0.45)
ax.set_axisbelow(True)
ax.bar_label(
    bars,
    labels=[f"{p:.1f}% (n={n})" for p, n in zip(plot_data["Percentage"], plot_data["Frequency"])],
    padding=4,
)
fig.tight_layout()
fig.savefig(image_path, dpi=300, bbox_inches="tight")
display(fig)
plt.close(fig)
print(f"Saved graphic to: {image_path}")
```

    Filtered cases with πρόσωπον: 14



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Greek lexical value after πρόσωπον</th>
      <th>Frequency</th>
      <th>Percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[missing]</td>
      <td>6</td>
      <td>42.9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>βασιλεύς</td>
      <td>2</td>
      <td>14.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Αγχους</td>
      <td>1</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Σαλωμών</td>
      <td>1</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>κύριος</td>
      <td>1</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>ἀλλόφυλος</td>
      <td>1</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Χαλδαῖος</td>
      <td>1</td>
      <td>7.1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>πῦρ</td>
      <td>1</td>
      <td>7.1</td>
    </tr>
  </tbody>
</table>
</div>



    
![png](TheValenceOfFear_HIPHIL_o.glanz_v20261110_files/TheValenceOfFear_HIPHIL_o.glanz_v20261110_119_2.png)
    


    Saved graphic to: article_images/006_Greek_Fear_Valence_goverened_nouns_withProsopon.png



```python

```


```python

```


```python

```
