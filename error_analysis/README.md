# Error analysis
Comparison of human annotation to model outputs on wrong test answers.

`roberta-large` fine-tuned with LlamBERT using the IMDB train data:

<table border="1">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" align="center">Human sentiment</th>
    </tr>
    <tr>
      <th>RoBERTa sentiment</th>
      <th>Positive</th>
      <th>Negative</th>
      <th>Mixed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Positive</td>
      <td align="center">31</td>
      <td align="center">16</td>
      <td align="center">13</td>
    </tr>
    <tr>
      <td align="center">Negative</td>
      <td align="center">17</td>
      <td align="center">14</td>
      <td align="center">9</td>
    </tr>
  </tbody>
</table>

`roberta-large` fine-tuned using the combined (extra+train) approach:

<table border="1">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" align="center">Human sentiment</th>
    </tr>
    <tr>
      <th>RoBERTa sentiment</th>
      <th>Positive</th>
      <th>Negative</th>
      <th>Mixed</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td align="center">Positive</td>
      <td align="center">25</td>
      <td align="center">17</td>
      <td align="center">13</td>
    </tr>
    <tr>
      <td align="center">Negative</td>
      <td align="center">15</td>
      <td align="center">14</td>
      <td align="center">16</td>
    </tr>
  </tbody>
</table>
