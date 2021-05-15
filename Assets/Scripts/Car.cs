using System.Collections;
using System.Collections.Generic;
using UnityEngine;


public class Car : MonoBehaviour
{
    [SerializeField] private float speed;
    [SerializeField] private float rotation;
    [SerializeField] private LayerMask raycastMask;
    [SerializeField] private int rayDistance = 10;

    private float[] input = new float[5];

    public NN network;

    private int carScore = 0;
    private bool collided;


    void FixedUpdate()
    {
        if (!collided)
        {
            for (int i = 0; i < 5; i++)
            {
                Vector3 newVector = Quaternion.AngleAxis(i * 45 - 90, new Vector3(0, 1, 0)) * transform.right;
                RaycastHit hit;
                Ray Ray = new Ray(transform.position, newVector);

                Debug.DrawRay(transform.position, newVector * rayDistance);

                if (Physics.Raycast(Ray, out hit, rayDistance, raycastMask))
                {
                    input[i] = (10 - hit.distance) / 10;
                }
            }

            float[] output = network.FeedForward(input);

            transform.Rotate(0, output[0] * rotation, 0, Space.World);
            transform.position += transform.right * output[1] * speed;
        }
    }


    void OnCollisionEnter(Collision collision)
    {
        if(collision.collider.gameObject.layer == LayerMask.NameToLayer("CheckPoint"))
        {
            GameObject[] checkPoints = GameObject.FindGameObjectsWithTag("CheckPoint");
            for (int i=0; i < checkPoints.Length; i++)
            {
                if(collision.collider.gameObject == checkPoints[i] && i == (carScore + 1 + checkPoints.Length) % checkPoints.Length)
                {
                    carScore++;
                    break;
                }
            }
        }
        else if(collision.collider.gameObject.layer != LayerMask.NameToLayer("Car"))
        {
            collided = true;
        }
    }


    public void UpdateScore()
    {
        network.UpdateScore(carScore);
    }
}
