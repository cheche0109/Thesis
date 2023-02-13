

def plot_component2(df_tmp, G_sub, savefig=False, **kwargs):
    #df_tree = pd.DataFram


    #print('# Number of clusters:', nx.number_connected_components(G_sub))
    #print('# Component size:', len(G_sub.nodes()))

    #df_tree = pd.concat([df_tree, write_df(G)], axis=0, ignore_index=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,8))
    nx.draw(G_sub, with_labels=True, node_size=3000, node_color='#2437e0', font_weight='bold', font_color='white', font_size=7, ax=ax1)
    ax1.set_title('Number of nodes: {}'.format(len(G_sub.nodes())))

    seq_len = df_tmp[['Query', 'Query_length']].drop_duplicates().set_index('Query').to_dict()['Query_length']
    sorted_node = df_tmp.sort_values('Query_length', ascending=False)['Query'].unique()
    df_tmp_pivot = pd.pivot_table(data=df_tmp, index='Query', columns='Subject', values='Evalue', aggfunc='first').fillna(11).reindex(index=sorted_node, columns=sorted_node)

    sns.heatmap(df_tmp_pivot, cmap='crest_r', annot=True, annot_kws={"fontsize":7}, ax=ax2)
    ax2.set_yticklabels([f'{i.get_text()}:{seq_len[i.get_text()]}' for i in ax2.axes.get_yticklabels()])

    # thres_list = [1e-20, 1e-30, 1e-40, 1e-50, 1e-60, 1e-70, 1e-80, 1e-90, 1e-100]
    # size_list = []
    # for thres in thres_list:
    #     rep_list = representative(df_tmp, thres)
    #     size_list.append(len(rep_list))
    # sns.lineplot(x=np.log10(thres_list), y=size_list, ax=ax4, markers='o')
    # ax4.set_xlabel('log10 Threshold')
    # ax4.set_ylabel('Number of representatives')


    if savefig:
        plt.tight_layout()
        plt.savefig(kwargs['filename'], dpi=1000)


    #return df_tmp

    # for idx, component in enumerate(nx.connected_components(G)):
    #     print(idx, len(component))
    #     node_list = sorted(list(component))
    #     print(node_list)
    #return df_tree