---
layout: page
title: Proceedings
permalink: /proceedings/
---


<table>
  <thead>
    <tr>
      <th>Sl. No</th>
      <th>Lecture </th>
      <th>Date </th>
    </tr>
  </thead>
  <tbody>
    {% for post in site.posts reversed %}
    <tr>
      <td>{{ forloop.index }}</td>
      <td><a href="{{ post.url | relative_url }}">{{ post.title }}</a></td>
      <td>{{ post.date | date: "%d-%B-%Y" }}</td>
    </tr>
    {% endfor %}
  </tbody>
</table>

