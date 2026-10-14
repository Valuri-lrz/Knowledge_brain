"""
pdf导入milvus流程图:
                                  +-----------+
                                  | __start__ |
                                  +-----------+
                                         *
                                         *
                                         *
                                  +------------+
                                  | entry_node |.
                            ......+------------+ .....
                       .....           .              .....
                 ......                .                   ....
              ...                     .                        .....
+----------------+                   ..                             ...
| pdf_to_md_node |                 ..                                 .
+----------------+               ..                                   .
               ***            ...                                     .
                  **        ..                                        .
                    **    ..                                          .
                +-------------+                                       .
                | md_img_node |                                       .
                +-------------+                                       .
                        *                                             .
                        *                                             .
                        *                                             .
            +---------------------+                                   .
            | document_split_node |                                   .
            +---------------------+                                   .
                        *                                             .
                        *                                             .
                        *                                             .
        +----------------------------+                                .
        | item_name_recognition_node |                                .
        +----------------------------+                                .
                        *                                             .
                        *                                             .
                        *                                             .
            +--------------------+                                    .
            | bge_embedding_node |                                    .
            +--------------------+                                    .
                        *                                             .
                        *                                             .
                        *                                             .
            +--------------------+                                  ...
            | import_milvus_node |                             .....
            +--------------------+                         ....
                               ***                    .....
                                  ***            .....
                                     **       ...
                                    +---------+
                                    | __end__ |
                                    +---------+
"""

import json

from langgraph.constants import END
from langgraph.graph.state import CompiledStateGraph, StateGraph

from knowledge.processor.import_process.nodes.bge_embedding import BgeEmbeddingChunksNode
from knowledge.processor.import_process.nodes.ducment_split import DocumentSplitNode
from knowledge.processor.import_process.nodes.entry import EntryNode
from knowledge.processor.import_process.nodes.import_milvus import ImportMilvusNode
from knowledge.processor.import_process.nodes.item_name_recognition import ItemNameRecognitionNode
from knowledge.processor.import_process.nodes.md_img import MarkDownImageNode
from knowledge.processor.import_process.nodes.pdf_to_md import PdfToMdNode
from knowledge.processor.import_process.state import ImportGraphState


def router(state: ImportGraphState) -> str:
    """根据状态返回对应的节点标识"""
    if state.get("is_pdf_read_enabled"):
        return "pdf"
    if state.get("is_md_read_enabled"):
        return "md"
    # raise ValueError("is_pdf_read_enabled或is_md_read_enabled必须有一个是True")
    return "END"


def create_import_graph() -> CompiledStateGraph:
    """
        创建导入流程图

    :return: CompiledStateGraph
    """

    # 1.创建
    graph = StateGraph(ImportGraphState)

    # 2.设置开启节点
    graph.set_entry_point("entry_node")

    # 3.增加节点
    nodes = {
        "entry_node": EntryNode(),
        "pdf_to_md_node": PdfToMdNode(),
        "md_img_node": MarkDownImageNode(),
        "document_split_node": DocumentSplitNode(),
        "item_name_recognition_node": ItemNameRecognitionNode(),
        "bge_embedding_node": BgeEmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }

    for name, node in nodes.items():
        graph.add_node(name, node)

    # 4.增加条件边
    graph.add_conditional_edges(
        "entry_node",
        router,  # 自动传状态对象
        {
            "pdf": "pdf_to_md_node",
            "md": "md_img_node",
            "END": END
        }
    )

    # 5.增加顺序边
    graph.add_edge("pdf_to_md_node", "md_img_node")
    graph.add_edge("md_img_node", "document_split_node")
    graph.add_edge("document_split_node", "item_name_recognition_node")
    graph.add_edge("item_name_recognition_node", "bge_embedding_node")
    graph.add_edge("bge_embedding_node", "import_milvus_node")
    graph.add_edge("import_milvus_node", END)

    # 6.编译
    return graph.compile()


kb_import_graph = create_import_graph()


def run_import_graph(import_file_path: str, file_dir: str) -> dict:
    """运行导入流程"""
    state = {
        "is_pdf_read_enabled": True,
        "is_md_read_enabled": False,
        "import_file_path":import_file_path,
        "file_dir":file_dir
    }

    final_state = None
    for event in kb_import_graph.stream(state):
        for name_node ,state in event.items():
            print(name_node)
            final_state = state
    return final_state


# ================================================================== #
#                        测试                                        #
# ================================================================== #
if __name__ == "__main__":
    #import_file_path = "E:\\doc\\万用表RS-12的使用.pdf"
    import_file_path = "E:\\doc\\H3C LA2608室内无线网关 用户手册-6W100-整本手册.pdf"
    file_dir = "E:\\temp_dir2"

    # 运行导入流程
    final_state = run_import_graph(import_file_path, file_dir)

    #print(json.dumps(final_state, indent=4, ensure_ascii=False))

    kb_import_graph.get_graph().print_ascii()


"""
{
    "is_md_read_enabled": false,
    "is_pdf_read_enabled": true,
    "import_file_path": "E:\\doc\\万用表RS-12的使用.pdf",
    "file_dir": "E:\\temp_dir"
}
"""