using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using System.IO;

public class NN : IComparable<NN>
{
    private float learningRate;
    private Layer[] layers;

    private int score = 0;

    public NN(int[] sizes)
    {
        //this.learningRate = learningRate;
        layers = new Layer[sizes.Length];
        for (int i = 0; i < sizes.Length; i++)
        {
            int nextSize = 0;
            if (i < sizes.Length - 1) nextSize = sizes[i + 1];
            layers[i] = new Layer(sizes[i], nextSize);
            for (int j = 0; j < sizes[i]; j++)
            {
                layers[i].biases[j] = UnityEngine.Random.Range(-1f, 1f);
                for (int k = 0; k < nextSize; k++)
                {
                    layers[i].weights[j, k] = UnityEngine.Random.Range(-1f, 1f);
                }
            }
        }
    }

    public int CompareTo(NN other)
    {
        if (other == null) return 1;

        if (score > other.score)
            return 1;
        else if (score < other.score)
            return -1;
        else
            return 0;
    }
    public void UpdateScore(int carScore)
    {
        score = carScore;
    }
    public float[] FeedForward(float[] inputs)
    {

        System.Array.Copy(inputs, 0, layers[0].neurons, 0, inputs.Length);
        for (int i = 1; i < layers.Length; i++)
        {
            Layer l = layers[i - 1];
            Layer l1 = layers[i];
            for (int j = 0; j < l1.size; j++)
            {
                float value = 0f;

                for (int k = 0; k < l.size; k++)
                {
                    value += l.neurons[k] * l.weights[k, j];
                }
                l1.neurons[j] = Sigmoid(value + l1.biases[j]);

            }
        }

        return layers[layers.Length - 1].neurons;
    }


    private float Sigmoid(float x)
    {
        return (float)Math.Tanh(x);
        //return 1 / (1 + Mathf.Exp(-(float)x));
    }

    private float Dsigmoid(float x)
    {
        return x * (1 - x);
    }

    public void BackPropagation(float[] targets)
    {
        float[] errors = new float[layers[layers.Length - 1].size];
        for (int i = 0; i < layers[layers.Length - 1].size; i++)
        {
            errors[i] = (targets[i] - layers[layers.Length - 1].neurons[i]);
        }
        for (int k = layers.Length - 2; k >= 0; k--)
        {
            Layer l = layers[k];
            Layer l1 = layers[k + 1];

            float[] errorsNext = new float[l.size];
            float[] gradients = new float[l1.size];
            for (int i = 0; i < l1.size; i++)
            {
                gradients[i] = errors[i] * Dsigmoid(layers[k + 1].neurons[i]);
                gradients[i] *= learningRate;
            }
            float[,] deltas = new float[l1.size, l.size];
            for (int i = 0; i < l1.size; i++)
            {
                for (int j = 0; j < l.size; j++)
                {
                    deltas[i, j] = gradients[i] * l.neurons[j];
                }
            }
            for (int i = 0; i < l.size; i++)
            {
                errorsNext[i] = 0;
                for (int j = 0; j < l1.size; j++)
                {
                    errorsNext[i] += l.weights[i, j] * errors[j];
                }
            }
            errors = new float[l.size];
            System.Array.Copy(errorsNext, 0, errors, 0, l.size);
            float[,] weightsNew = new float[l.weights.GetLength(0), l.weights.GetLength(1)];
            for (int i = 0; i < l1.size; i++)
            {
                for (int j = 0; j < l.size; j++)
                {
                    weightsNew[j, i] = l.weights[j, i] + deltas[i, j];
                }
            }
            l.weights = weightsNew;
            for (int i = 0; i < l1.size; i++)
            {
                l1.biases[i] += gradients[i];
            }
        }
    }

    public void Mutate(int chance, float val)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            int nextSize = 0;
            if (i < layers.Length - 1) nextSize = layers[i + 1].size;
            for (int j = 0; j < layers[i].size; j++)
            {
                layers[i].biases[j] = (UnityEngine.Random.Range(0f, chance) <= 5) ? layers[i].biases[j] += UnityEngine.Random.Range(-val, val) : layers[i].biases[j];
                for (int k = 0; k < nextSize; k++)
                {
                    layers[i].weights[j, k] = (UnityEngine.Random.Range(0f, chance) <= 5) ? layers[i].weights[j, k] += UnityEngine.Random.Range(-val, val) : layers[i].weights[j, k];
                }
            }
        }
    }

    public NN DeepCopyNetwork(NN nn)
    {
        for (int i = 0; i < layers.Length; i++)
        {
            int nextSize = 0;
            if (i < layers.Length - 1) nextSize = layers[i + 1].size;
            for (int j = 0; j < layers[i].size; j++)
            {
                nn.layers[i].biases[j] = layers[i].biases[j];
                for (int k = 0; k < nextSize; k++)
                {
                    nn.layers[i].weights[j, k] = layers[i].weights[j, k];
                }
            }
        }
        return nn;
    }
    public void Load(string path)
    {
        TextReader tr = new StreamReader(path);
        int NumberOfLines = (int)new FileInfo(path).Length;
        string[] ListLines = new string[NumberOfLines];
        int index = 1;
        for (int i = 1; i < NumberOfLines; i++)
        {
            ListLines[i] = tr.ReadLine();
        }
        tr.Close();
        if (new FileInfo(path).Length > 0)
        {
            for (int i = 0; i < layers.Length; i++)
            {
                int nextSize = 0;
                if (i < layers.Length - 1) nextSize = layers[i + 1].size;
                for (int j = 0; j < layers[i].size; j++)
                {
                    layers[i].biases[j] = float.Parse(ListLines[index]);
                    index++;
                    for (int k = 0; k < nextSize; k++)
                    {
                        layers[i].weights[j, k] = float.Parse(ListLines[index]);
                        index++;
                    }
                }
            }
            
        }
    }
    public void Save(string path)
    {
        File.Create(path).Close();
        StreamWriter writer = new StreamWriter(path, true);

        for (int i = 0; i < layers.Length; i++)
        {
            int nextSize = 0;
            if (i < layers.Length - 1) nextSize = layers[i + 1].size;
            for (int j = 0; j < layers[i].size; j++)
            {
                writer.WriteLine(layers[i].biases[j]);
                for (int k = 0; k < nextSize; k++)
                {
                    writer.WriteLine(layers[i].weights[j, k]);
                }
            }
        }
        writer.Close();
    }
}

public class Layer
{
    public int size;
    public float[] neurons;
    public float[] biases;
    public float[,] weights;

    public Layer(int size, int nextSize)
    {
        this.size = size;
        neurons = new float[size];
        biases = new float[size];
        weights = new float[size, nextSize];
    }
}
