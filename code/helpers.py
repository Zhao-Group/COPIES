import pandas as pd
def import_gff_file(gff):
    '''
    imports and formats a genome annotation file in the GFF format for use
    '''
    start_index = 0
    with open(gff,"r") as raw_gff:
        #if args.verbose:
        print(f'Annotation file {gff} successfully imported')
        gff_lines = raw_gff.readlines()
        for index in range(len(gff_lines)):
            if ("##" not in gff_lines[index]):
                start_index = index
                break
    col_names = ["chromosome", "source", "type", "Start", "Stop", "score", "Strand", "phase", "qualifiers"]
    
    ### Column names used in main.py
    ### gene_table = total_gene_table[total_gene_table['assembly_unit'] == 'Primary Assembly'][['# feature','class','chromosome','genomic_accession','start','end','strand','locus_tag','product_accession']].reset_index(drop=True)
    ### gene_table.columns = ['# feature','class','#Name','Accession', 'Start', 'Stop', 'Strand', 'Locus tag','Protein product']
    #### refined_gene_table = gene_table[gene_table['# feature']=='gene'][['Accession', 'Start', 'Stop', 'Strand', 'Locus tag']].reset_index(drop=True)

    gff_df = pd.read_csv(gff, sep='\t', skiprows = start_index, header = None, names = col_names)
    #if args.verbose:
    print(f'Annotation database successfully generated')
    return gff_df

def gff3_to_gene_table(gff_df):
    """
    Parses a GFF3 file and extracts gene information to create a gene_table dataframe.

    Args:
        gff_file (str): Path to the input GFF3 data_frame.
    
    Returns
        gt_df (pd.Dataframe): Dataframe containing the gene table information in relevant format.
    """

    gt_df = pd.DataFrame(columns=['seq_id','start','end','strand','locus_tag','gene_name','product'])
    new_data = []

    for index,feature in gff_df.iterrows():
        if feature.type == "gene":
            # Extract relevant information from the gene feature
            seq_id = feature.qualifiers.split(";")[0] #  # or feature.qualifiers.get("seqid", [""])[0] if available
            start = int(feature.Start) #.location.start + 1  # GFF3 is 1-indexed
            end = int(feature.Stop) #.location.end
            strand = feature.Strand
                    
            # Extract locus_tag, gene_name, and product from qualifiers
            locus_tag = feature.qualifiers.split(";")[1]#.get("locus_tag", [""])[0]
            gene_name = feature.qualifiers.split(";")[2]#.get("gene", [""])[0]  # Use "gene" as per some examples
            product = feature.qualifiers.split(";")[3]#.get("product", [""])[0]
                    
            # Append the extracted information to post to the gene table
            new_data.append([seq_id,start,end,strand,locus_tag,gene_name,product])

    # Add the row to the dataframe
    new_df = pd.DataFrame(new_data,columns=['seq_id','start','end','strand','locus_tag','gene_name','product'])
    gt_df = pd.concat([gt_df, new_df], ignore_index=True)
    return gt_df