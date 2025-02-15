<h1>SEACOP: Using Machine Learning and Underwater Submersibles to Monitor Overfishing and Bycatch</h1>
<h2> Problem </h2>
Overfishing refers to the practice of catching fish at a rate faster than they can naturally reproduce, leading to the depletion of fish populations. This practice threatens the sustainability of fisheries and marine ecosystems, as it can cause species populations to collapse and disrupt the balance of the marine food web. The decline in fish stocks has far-reaching ecological, economic, and social consequences, affecting biodiversity, ecosystem health, and livelihoods dependent on fisheries.

Bycatch refers to the unintentional capture of non-target species during fishing operations. These species can include marine mammals, turtles, seabirds, sharks, and juvenile fish of commercial species. Bycatch is often discarded, leading to the waste of marine life and contributing to the decline of vulnerable species. Non-selective fishing gear, such as trawls and longlines, is a major cause of bycatch, which poses a significant threat to marine biodiversity and ecosystems.

Overfishing poses significant problems for both marine ecosystems and human societies. Here are some of the primary issues identified by research:

1. Depletion of Fish Stocks : Overfishing directly leads to the depletion of fish stocks, driving many species to dangerously low levels or even collapse, which can devastate ecosystems and economies dependent on fisheries Fish, Markets, and Fishermen: The Economics of Overfishing: Iudicello, Suzanne, Weber, Michael L., Wieland, Robert: 9781559636438: Amazon.Com: Books, www.amazon.com/Fish-Markets-Fishermen-Economics-Overfishing/dp/1559636432. Accessed 15 Oct. 2024.

2. Destruction of Marine Habitats: Overfishing, particularly with destructive fishing gear such as mobile bottom trawls, destroys important marine habitats, reducing biodiversity and affecting the overall health of marine ecosystems 

3. Ecosystem Imbalance: Removing large predatory fish can cause trophic cascades, disrupting the balance of marine food webs. This leads to unintended consequences such as the overgrowth of algae or collapse of other species, which further reduces biodiversity. 
4. Economic Impacts: Overfishing leads to the economic decline of fishing communities, especially when key species become too scarce to harvest sustainably. This results in a loss of livelihood for those dependent on the fishing industry. 

5. Difficulties in Fishery Management: Political and economic incentives often complicate efforts to reduce fishing pressure, even when there is clear scientific evidence of overexploitation. This results in slow policy response and continued resource depletion.

<h2> Proposed Solution </h2>

SEACOP Aims to provide a Hollistic End to End Monitoring Solution for overfishing. The first step utilizes Underwater Submersibles to record video footage of fishing nets. This footage is then analysed by an Object Tracker which will track all the fish(Target Catch), Sharks(Bycatch), and Turtles(Bycatch) present in the video. Moving objects are excluded from the final report as they are likley not actually caught in the fishing nets. This monitoring solution will enable reuglators to better enforce fishing regulations as current methods rely on Cooperation of Fishermen or the use of Fisheries observes, both of which can be unreliable especially when dealing with IUU(Illegal, Underreported, and Unregulated) fishing.
  
<h2>Contents of the Repository</h2>
This repo contains my customized implementation of the DeepSORT Algorithm. The algorithm uses a YOLO11 model I trained on a custom Dataset to identify Fish, Sharks and Turtles. By levaraging Object Tracking this algorithm is able to monitor Catch Size and Bycatch. The program also uses Compares the intersection over union of each object found in the video as the video frames progress. By graphing IoU over time the program is able to classify if the objects are moving or if they are still. This functionality is to help the program decide if the object is caught in a finshing net or not. 

The Dataset used to train the YOLO Object detector was custom curated and custom annotated, sources of the raw images are listed in the citations. 
Here is the Link to the Dataset I made for this project: https://www.kaggle.com/datasets/akanshkarthik/overfishing-object-detection-dataset

<strong>Citations</strong>

Incorporated, Bajan Digital Creations. “Blue Bot Dataset: Atlantic Tang.” Kaggle, 6 Nov. 2020, www.kaggle.com/datasets/hiyaro/atlantictang.

Incorporated, B. D. C. (2020, November 6). Blue Bot Dataset: Fish body plans. Kaggle. https://www.kaggle.com/datasets/hiyaro/fishbodyplans

Lautaro. (2020, October 30). Shark species. Kaggle. https://www.kaggle.com/datasets/larusso94/shark-species/data

Wildlifedatasets. “SEATURTLEID2022.” Kaggle, 7 Feb. 2024, www.kaggle.com/datasets/wildlifedatasets/seaturtleid2022.

@inproceedings{Wojke2017simple,
  title={Simple Online and Realtime Tracking with a Deep Association Metric},
  author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
  booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
  year={2017},
  pages={3645--3649},
  organization={IEEE},
  doi={10.1109/ICIP.2017.8296962}
}

@inproceedings{Wojke2018deep,
  title={Deep Cosine Metric Learning for Person Re-identification},
  author={Wojke, Nicolai and Bewley, Alex},
  booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
  year={2018},
  pages={748--756},
  organization={IEEE},
  doi={10.1109/WACV.2018.00087}
}
