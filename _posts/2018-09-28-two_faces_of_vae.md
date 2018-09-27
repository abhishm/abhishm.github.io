---
layout: post
title: Understanding Variational Auto-Encoder
author: "Abhishek Mishra"
---
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
<style>
.center-image
{
    margin: 0 auto;
    display: block;
}
</style>

# Variational Auto Encoder (VAE)

This is a post to understand why VAE works. I will provide two perspectives on the working of VAE. One is the `generative perspective` in which VAE is used as a tool to generate data samples from noise. The other is `encoding-decoding perspective` in which we purposefully make the life of encoder and decoder difficult so that they can do their jobs better. Specifically, encoder can encode data samples well and decoder can decode the data samples from encoded signal well.
