using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Manager : MonoBehaviour
{
    [SerializeField, Range(1, 60)] private int timeframe;

    [SerializeField, Range(1, 100)] private int populationSize;
    
    [SerializeField, Range(0f, 1f)] private float MutationChance = 0.01f;

    [SerializeField, Range(0f, 1f)] private float MutationStrength = 0.5f;

    [SerializeField, Range(1, 10)] private int Gamespeed = 1;

    [SerializeField] private GameObject prefab;

    private int[] layers = new int[3] { 5, 3, 2 };
    private List<NN> networks;
    private List<Car> cars;

    void Start()
    {
        InitNetworks();
        InvokeRepeating("CreateCars", 0.1f, timeframe);
    }

    public void InitNetworks()
    {
        networks = new List<NN>();
        for (int i = 0; i < populationSize; i++)
        {
            NN net = new NN(layers);
            net.Load("Assets/Saves/Save.txt");
            networks.Add(net);
        }
    }

    public void CreateCars()
    {
        Time.timeScale = Gamespeed;
        if (cars != null)
        {
            for (int i = 0; i < cars.Count; i++)
            {
                GameObject.Destroy(cars[i].gameObject);
            }
            SortNetworks();
        }

        cars = new List<Car>();
        for (int i = 0; i < populationSize; i++)
        {
            Car car = (Instantiate(prefab, new Vector3(0, 1.6f, -16), new Quaternion(0, 0, 1, 0))).GetComponent<Car>();
            car.network = networks[i];
            cars.Add(car);
        }
        
    }

    public void SortNetworks()
    {
        for (int i = 0; i < populationSize; i++)
        {
            cars[i].UpdateScore();
        }
        networks.Sort();
        networks[populationSize - 1].Save("Assets/Pre-trained.txt");
        //networks[populationSize - 1].Save("Assets/Saves/Save.txt");
        for (int i = 0; i < populationSize / 2; i++)
        {
            networks[i] = networks[i + populationSize / 2].DeepCopyNetwork(new NN(layers));
            
            networks[i].Mutate((int)(1/MutationChance), MutationStrength);
        }
    }
}
