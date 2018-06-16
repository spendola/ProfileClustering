import os
import urllib
import urllib2
import xml.etree.ElementTree
import time
import numpy as np
import random

def main():
    print("Profile Clustering\nversion 0.1\n")
    xml = None
    batch_id = None

    while(True):
        print(" ")
        print("1) Evaluate Dataset")
        print("2) Generate Footprints")
        print("3) Footprint Clustering")
        print("4) Adaptive Clustering")
        print("5) Profile Labeling")
        print("6) Full Cycle")
        print("8) Get Batch Id")
        print("9) Download Profile Markings")
        print("0) Exit")

        choice = int(raw_input("Select an Option: "))
        if(choice == 1):
            dataset, profiles = GetDataset(xml)
        if(choice == 2):
            DownloadFile("http://www.pendola.net/tools/export_profile_markings.php", "profile_markings.txt")
            TouchFile("http://www.pendola.net/tools/export_set.php")
            DownloadFile("http://www.pendola.net/tools/dump.txt", "dump.txt")
            dataset, profiles = GetDataset(xml)
            markings, weights = GetProfileMarkings()
            GenerateFootprints(dataset, profiles, markings, weights)

        if(choice == 3):
            TouchFile("http://www.pendola.net/tools/export_profile_footprints.php")
            DownloadFile("http://www.pendola.net/tools/profile_footprints.txt", "profile_footprints.txt")
            footprints = GetProfileFootprints()
            FootprintClustering(footprints, batch_id)

        if(choice == 4):
            DownloadFile("http://www.pendola.net/tools/export_profile_markings.php", "profile_markings.txt")
            TouchFile("http://www.pendola.net/tools/export_set.php")
            DownloadFile("http://www.pendola.net/tools/dump.txt", "dump.txt")
            dataset, profiles = GetDataset(xml)
            best_bias = 0
            lowest_density = None
            temp = 1
            for i in range(25):
                for e in range(2):
                    temp = tem * (-1)
                    print("Attempting Bias : " + str(best_bias + temp))
                    markings, weights = GetProfileMarkings(best_bias + temp)
                    GenerateFootprints(dataset, profiles, markings, weights)

                    TouchFile("http://www.pendola.net/tools/export_profile_footprints.php")
                    DownloadFile("http://www.pendola.net/tools/profile_footprints.txt", "profile_footprints.txt")
                    footprints = GetProfileFootprints()
                    for e in range(3):
                        this_density = FootprintClustering(footprints, batch_id)
                        if(lowest_density == None):
                            lowest_density = this_density
                            best_bias = i
                        if(this_density < lowest_density):
                            lowest_density = this_density
                            best_bias = i
                
            print "Lowest Density: " + str(lowest_density)
            print "Best Bias: " + str(best_bias)
                    
            
        if(choice == 5):
            TouchFile("http://www.pendola.net/tools/export_profile_clusters.php")
            DownloadFile("http://www.pendola.net/tools/profile_clusters.txt", "profile_clusters.txt")
            TouchFile("http://www.pendola.net/tools/export_profile_footprints.php")
            DownloadFile("http://www.pendola.net/tools/profile_footprints.txt", "profile_footprints.txt")
            clusters = GetProfileClusters()
            footprints = GetProfileFootprintsEx()
            ProfileLabeling(footprints, clusters)
            
        if(choice == 6):
            # Generate Footprints for all profiles in database dump
            DownloadFile("http://www.pendola.net/tools/export_profile_markings.php", "profile_markings.txt")
            TouchFile("http://www.pendola.net/tools/export_set.php")
            DownloadFile("http://www.pendola.net/tools/dump.txt", "dump.txt")
            dataset, profiles = GetDataset(xml)
            markings, weights = GetProfileMarkings()
            GenerateFootprints(dataset, profiles, markings, weights)

            # Generate footprint clusters
            batch_id = FetchRemoteBatchId()
            TouchFile("http://www.pendola.net/tools/export_profile_footprints.php")
            DownloadFile("http://www.pendola.net/tools/profile_footprints.txt", "profile_footprints.txt")
            footprints = GetProfileFootprints()
            FootprintClustering(footprints, batch_id)

            # Label all profiles
            TouchFile("http://www.pendola.net/tools/export_profile_clusters.php")
            DownloadFile("http://www.pendola.net/tools/profile_clusters.txt", "profile_clusters.txt")
            TouchFile("http://www.pendola.net/tools/export_profile_footprints.php")
            DownloadFile("http://www.pendola.net/tools/profile_footprints.txt", "profile_footprints.txt")
            clusters = GetProfileClusters()
            footprints = GetProfileFootprintsEx()
            ProfileLabeling(footprints, clusters)
            


        if(choice == 8):
            batch_id = FetchRemoteBatchId()
            print("Batch ID: " + batch_id)
        if(choice == 9):
            DownloadFile("http://www.pendola.net/tools/export_profile_markings.php", "profile_markings.txt")
        if(choice == 0):
            break

        
def ProfileLabeling(profiles, clusters):
    for profile in profiles:
        best_distance = None
        best_cluster = None
        for cluster in clusters:
            this_distance = np.linalg.norm(np.asarray(profile[1])-np.asarray(cluster[1]))
            if(best_distance == None):
                best_distance = this_distance
                best_cluster = cluster[0]
            if(this_distance < best_distance):
                best_distance = this_distance
                best_cluster = cluster[0]
        print "Best Cluster (" + profile[0] + ") " + str(best_cluster)
        sql = "UPDATE profile_footprints SET label='"+ str(best_cluster) +"' WHERE profile='"+profile[0]+"'"
        UpdateRemoteAPI(sql, "raw_sql")
                
    
def FootprintClustering(footprints, batch_id, cluster_count=6):
    vectors, maxima = VectorizeFootprints(footprints)

    clusters = []
    for c in range(cluster_count):
        index = random.randint(1, len(vectors)-1)
        vectorf = np.zeros((len(vectors[0][1])), dtype=np.float)
        for i in range(len(vectors[0][1])):
            vectorf[i] = vectors[index][1][i]
        clusters.append(vectorf)
        
    while(True):
        # calculate distances
        for vector in vectors:
            best_distance = None
            for i in range(len(clusters)):
                this_distance = np.linalg.norm(vector[1]-clusters[i])
                if(best_distance == None):
                    best_distance = this_distance
                    vector[0] = i
                if(this_distance < best_distance):
                    best_distance = this_distance
                    vector[0] = i
            
        # recalculate clusters
        variance = 0.0
        for c in range(len(clusters)):
            for i in range(len(clusters[0])):
                total = 0.0
                count = 0
                
                for vector in vectors:
                    if(vector[0] == c):
                        total = total + vector[1][i]
                        count = count + 1

                new_position = max(float(total) / max(float(count), 1), 0.00000)
                variance = variance + (abs(new_position - clusters[c][i]))
                clusters[c][i] = new_position
        print str(variance) + " variance"

        highest_density = 0
        if(variance < 0.0001):
            print("Final Cluster Density: ")
            for i in range(len(clusters)):
                this_density = len([x for x in vectors if x[0] == i])
                if(this_density > highest_density):
                    highest_density = this_density
                print this_density,
            print " "

            print("Clusters")
            np.set_printoptions(suppress=True)
            for cluster in clusters:
                footprint = (np.array_str(cluster, precision=3)).replace("[", "").replace("]", "")
                footprint = ', '.join(footprint.split())
                print footprint
                if(batch_id != None):
                    sql = "INSERT INTO profile_clusters (batch_id, footprint, label, count) "
                    sql = sql + "VALUES (" + batch_id + ", '" + footprint + "', '', 0)"
                    UpdateRemoteAPI(sql, "raw_sql")
            break
    return highest_density
    

def VectorizeFootprints(footprints):           
    counter = 0
    maximas = {}
    markings = []
    sizes = {}
    
    for footprint in footprints:
        vector = footprint[1].split(",")
        vectorf = np.zeros((len(vector)), dtype=np.float)
        for i in range(len(vector)):

            # replace empty vectors by 0
            if(vector[i] == ""):
                vector[i] = "0.0"

            # vectorize as list of floats
            vectorf[i] = float(vector[i])

            # update the maxima
            if(i not in maximas):
                maximas[i] = 0.0
            if(vectorf[i] > maximas[i]):
                maximas[i] = vectorf[i]
                
        markings.append([0, vectorf])

    print(str(len(markings)) + " vectors created")
    return markings, maximas

def GenerateFootprints(dataset, profiles, markings, weights, lower_limit=25, upper_limit=50):
    counts = {}
    for profile in profiles:
       
        # Get a pile of phrases from this user
        phrases = []
        counter = 0
        for item in dataset:
            if item[0] == profile:
                counter = counter + 1
                if(counter > upper_limit):
                    break
                phrases.append(item[1])

        scores = ""
        union = ""
        caps = {}
        for family in markings:
            family_counter = 0.0
            for mark in family[1]:
                for phrase in phrases:
                    if mark in phrase:

                        if(mark not in caps):
                            caps[mark] = weights[mark]
                        else:
                            caps[mark] = caps[mark] + weights[mark]
                        if(caps[mark] < 1.0):
                            family_counter = family_counter + weights[mark]
                        
                        if(mark not in counts):
                            counts[mark] = 1
                        else:
                            counts[mark] = counts[mark] + 1
            scores = scores + union + str(round(family_counter, 5))
            union = ","

        if(counter > lower_limit):
            print(profile + ": " + scores)
            phrase = profile + " " + scores
            UpdateRemoteAPI(phrase, "publish_profile_footprint")

    weights_phrase = ""
    for weight in weights:
        if weight not in counts:
            print(weight + " not in counts")
        else:
            phrase = "UPDATE profile_markings SET value=" + str(counts[weight]) + " WHERE marking='" + weight[1:] + "'"
            UpdateRemoteAPI(phrase, "raw_sql")      

def GetDataset(tree):
    print("Processing Profile List ... ")
    if tree == None:
        tree = LoadFromXml("dump.txt")
        
    root = tree.getroot()
    total = len(root)
    percentage = 0.0

    dataset = []
    profiles = {}
    counter = 0
    for node in root:
        if(node[1].text != None and node[10].text != None):
            if node[1].text in profiles:
                profiles[node[1].text] = profiles[node[1].text] + 1
            else:
                profiles[node[1].text] = 1
            dataset.append([node[1].text, node[10].text])
        counter = counter + 1
        percentage = CalculatePercentage(counter, total, percentage)
        
    counter, subcounter, exactcounter = 0, 0, 0
    for profile in profiles:
        if profiles[profile] >= 20:
            counter = counter + 1
        elif profiles[profile] >= 15:
            subcounter = subcounter + 1
            
    print(str(counter) + " profiles are large enough for analysis")
    print(str(subcounter) + " profiles are almost large enough for analysis")
    return dataset, profiles

def GetProfileClusters():
    print("Processing Profile Clusters ... ")
    tree = LoadFromXml("profile_clusters.txt")
    root = tree.getroot()

    clusters = []
    for node in root:
        clusters.append([node[0].text, [float(x) for x in node[1].text.split(",")]])
    return clusters

def GetProfileMarkings(bias=0):
    print("Processing Profile Markings ... ")
    tree = LoadFromXml("profile_markings.txt")
    root = tree.getroot()

    markings = []
    weights = {}
    for node in root:
        marks = []
        family = node.attrib.get('name')
        for subnode in node:
            if(subnode[0].text == None or subnode[1].text == None):
                continue
            marks.append(" " + subnode[0].text)
            weights[" " + subnode[0].text] = max(np.exp(-(float(subnode[1].text + bias)) /10.0), 0.005)
        markings.append([family, marks])

    return markings, weights

def GetProfileFootprints():
    print("Processing Profile Footprints ... ")
    tree = LoadFromXml("profile_footprints.txt")
    root = tree.getroot()

    footprints = []
    for node in root:
        if(node[0].text != None and node[1].text != None):
            footprints.append([node[0].text, node[1].text])

    print(str(len(footprints)) + " footprints loaded")
    return footprints

def GetProfileFootprintsEx():
    print("Processing Profile Footprints ... ")
    tree = LoadFromXml("profile_footprints.txt")
    root = tree.getroot()

    footprints = []
    for node in root:
        if(node[0].text != None and node[1].text != None):
            text = node[1].text
            if(text[-1:] == ","):
                text = text[:-1]
            footprints.append([node[0].text, [float(x) for x in text.split(",")]])

    print(str(len(footprints)) + " footprints loaded")
    return footprints

def LoadFromXml(filename):
    print("loading " + filename)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    e = xml.etree.ElementTree.parse(os.path.join(dir_path, filename))
    return e

def CalculatePercentage(count, total, percentage):
    new_percentage = round((float(count)/float(total))*100.0, 1)
    if(percentage != new_percentage):
        print str(new_percentage) + "%"
    return new_percentage

def FetchRemoteBatchId():
    query_args = { "action": "fetch"}
    data = urllib.urlencode(query_args)
    try:
        response = urllib2.urlopen('http://www.pendola.net/api/fetch_profile_clusters_batchid.php', data)
        html = response.read()
        return html
    except:
        print "Exception"
        return -1

def TouchFile(remote_name):
    print("touching url: " + remote_name)
    response = urllib2.urlopen(remote_name)
    html = response.read()
    if html != None:
        print("touch successful")
    else:
        print("unable to determine touch response")

def DownloadFile(remote_name, local_name):
    print("downloading file from " + remote_name)
    response = urllib2.urlopen(remote_name)
    html = response.read()
    html = html.replace("&nbsp;", " ")
    dir_path = os.path.dirname(os.path.realpath(__file__))
    file = open(os.path.join(dir_path, local_name), "w+")
    file.write(html)
    file.close()
    print("download finished")

def UpdateRemoteAPI(msg, action="public_message"):
    # Prepare the data
    clean_msg = msg.encode('utf-8').strip()
    query_args = { 'action':action, 'message':clean_msg }
    data = urllib.urlencode(query_args)
    try:
        response = urllib2.urlopen('http://www.pendola.net/api/remote_status.php', data)
        html = response.read()
        return html
    except:
        print "Exception"
        return "exception"

if __name__ == '__main__':
    main()

