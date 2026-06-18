/* Copyright (c) 2026, Oracle and/or its affiliates.

   This program is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License, version 2.0,
   as published by the Free Software Foundation.

   This program is designed to work with certain software (including
   but not limited to OpenSSL) that is licensed under separate terms,
   as designated in a particular file or component or in included license
   documentation.  The authors of MySQL hereby grant you an additional
   permission to link the program and your derivative works with the
   separately licensed software that they have either included with
   the program or referenced in the documentation.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License, version 2.0, for more details.

   You should have received a copy of the GNU General Public License
   along with this program; if not, write to the Free Software
   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301  USA */

#ifndef ITEM_VECTORFUNC_INCLUDED
#define ITEM_VECTORFUNC_INCLUDED

#include <string>
#include <vector>

#include "sql/item_func.h"
#include "sql/item_strfunc.h"
#include "sql/parse_location.h"  // POS
#include "sql_string.h"

class THD;
class PT_item_list;

namespace vector_index_registry {
struct index_info;
}

namespace vector_index {
struct search_result;
}

class Item_func_vec_fromtext final : public Item_str_func {
  String m_value;

 public:
  Item_func_vec_fromtext(const POS &pos, Item *a) : Item_str_func(pos, a) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_fromtext"; }
};

class Item_func_vec_totext final : public Item_str_func {
  String m_value;
  String m_number_buf;

 public:
  Item_func_vec_totext(const POS &pos, Item *a) : Item_str_func(pos, a) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_totext"; }
};

class Item_func_vec_normalize final : public Item_str_func {
  String m_value;

 public:
  Item_func_vec_normalize(const POS &pos, Item *a) : Item_str_func(pos, a) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_normalize"; }
};

class Item_func_vector_dim final : public Item_int_func {
 public:
  Item_func_vector_dim(const POS &pos, Item *a) : Item_int_func(pos, a) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vector_dim"; }
};

class Item_func_vec_dot_product final : public Item_real_func {
 public:
  Item_func_vec_dot_product(const POS &pos, Item *a, Item *b)
      : Item_real_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  double val_real() override;
  const char *func_name() const override { return "vec_dot_product"; }
};

class Item_func_vec_inner_product final : public Item_real_func {
 public:
  Item_func_vec_inner_product(const POS &pos, Item *a, Item *b)
      : Item_real_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  double val_real() override;
  const char *func_name() const override { return "vec_inner_product"; }
};

class Item_func_vec_distance_euclidean final : public Item_real_func {
 public:
  Item_func_vec_distance_euclidean(const POS &pos, Item *a, Item *b)
      : Item_real_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  double val_real() override;
  const char *func_name() const override { return "vec_distance_euclidean"; }
};

class Item_func_vec_distance_cosine final : public Item_real_func {
 public:
  Item_func_vec_distance_cosine(const POS &pos, Item *a, Item *b)
      : Item_real_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  double val_real() override;
  const char *func_name() const override { return "vec_distance_cosine"; }
};

class Item_func_vec_distance final : public Item_real_func {
 public:
  Item_func_vec_distance(const POS &pos, PT_item_list *arg_list)
      : Item_real_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  double val_real() override;
  const char *func_name() const override { return "vec_distance"; }
};

class Item_func_vec_index_create final : public Item_int_func {
 public:
  Item_func_vec_index_create(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_create"; }
};

class Item_func_vec_index_drop final : public Item_int_func {
 public:
  Item_func_vec_index_drop(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_drop"; }
};

class Item_func_vec_index_rebuild final : public Item_int_func {
 public:
  Item_func_vec_index_rebuild(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_rebuild"; }
};

class Item_func_vec_index_bulk_load_begin final : public Item_int_func {
 public:
  Item_func_vec_index_bulk_load_begin(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_bulk_load_begin";
  }
};

class Item_func_vec_index_bulk_build final : public Item_int_func {
 public:
  Item_func_vec_index_bulk_build(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_bulk_build"; }
};

class Item_func_vec_index_recover final : public Item_int_func {
 public:
  Item_func_vec_index_recover(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_recover"; }
};

class Item_func_vec_index_rebuild_all final : public Item_int_func {
 public:
  Item_func_vec_index_rebuild_all(const POS &pos) : Item_int_func(pos) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_rebuild_all"; }
};

class Item_func_vec_index_recover_all final : public Item_int_func {
 public:
  Item_func_vec_index_recover_all(const POS &pos) : Item_int_func(pos) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_recover_all"; }
};

class Item_func_vec_index_info final : public Item_str_func {
  String m_value;
  String m_number_buf;

 public:
  Item_func_vec_index_info(const POS &pos, PT_item_list *arg_list)
      : Item_str_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_index_info"; }
};

class Item_func_vec_index_set_search_ef final : public Item_int_func {
 public:
  Item_func_vec_index_set_search_ef(const POS &pos, Item *a, Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_set_search_ef"; }
};

class Item_func_vec_index_set_hnsw_build_params final : public Item_int_func {
 public:
  Item_func_vec_index_set_hnsw_build_params(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_hnsw_build_params";
  }
};

class Item_func_vec_index_set_faiss_ivf_params final : public Item_int_func {
 public:
  Item_func_vec_index_set_faiss_ivf_params(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_faiss_ivf_params";
  }
};

class Item_func_vec_index_set_faiss_ivfpq_params final : public Item_int_func {
 public:
  Item_func_vec_index_set_faiss_ivfpq_params(const POS &pos,
                                             PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_faiss_ivfpq_params";
  }
};

class Item_func_vec_index_set_diskann_build_params final : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_build_params(const POS &pos,
                                               PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_build_params";
  }
};

class Item_func_vec_index_set_diskann_search_complexity final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_search_complexity(const POS &pos, Item *a,
                                                    Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_search_complexity";
  }
};

class Item_func_vec_index_set_diskann_search_beamwidth final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_search_beamwidth(const POS &pos, Item *a,
                                                   Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_search_beamwidth";
  }
};

class Item_func_vec_index_set_diskann_pq_code_budget_size final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_pq_code_budget_size(const POS &pos, Item *a,
                                                      Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_pq_code_budget_size";
  }
};

class Item_func_vec_index_set_diskann_disk_pq_dims final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_disk_pq_dims(const POS &pos, Item *a,
                                               Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_disk_pq_dims";
  }
};

class Item_func_vec_index_set_diskann_accelerate_build final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_accelerate_build(const POS &pos, Item *a,
                                                   Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_accelerate_build";
  }
};

class Item_func_vec_index_set_diskann_shuffle_build final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_shuffle_build(const POS &pos, Item *a,
                                                Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_shuffle_build";
  }
};

class Item_func_vec_index_set_diskann_use_bfs_cache final
    : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_use_bfs_cache(const POS &pos, Item *a,
                                                Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_use_bfs_cache";
  }
};

class Item_func_vec_index_set_diskann_build_mode final : public Item_int_func {
 public:
  Item_func_vec_index_set_diskann_build_mode(const POS &pos, Item *a, Item *b)
      : Item_int_func(pos, a, b) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_set_diskann_build_mode";
  }
};

class Item_func_vec_index_list final : public Item_str_func {
  String m_value;

 public:
  Item_func_vec_index_list(const POS &pos) : Item_str_func(pos) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_index_list"; }
};

class Item_func_vec_index_upsert final : public Item_int_func {
 public:
  Item_func_vec_index_upsert(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_upsert"; }
};

class Item_func_vec_index_upsert_batch final : public Item_int_func {
 public:
  Item_func_vec_index_upsert_batch(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_upsert_batch"; }
};

class Item_func_vec_index_erase final : public Item_int_func {
 public:
  Item_func_vec_index_erase(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_erase"; }
};

class Item_func_vec_index_search final : public Item_str_func {
  String m_value;
  String m_number_buf;

 public:
  Item_func_vec_index_search(const POS &pos, PT_item_list *arg_list)
      : Item_str_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_index_search"; }
};

class Item_func_vec_index_search_batch final : public Item_str_func {
  String m_value;
  String m_number_buf;

 public:
  Item_func_vec_index_search_batch(const POS &pos, PT_item_list *arg_list)
      : Item_str_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override { return "vec_index_search_batch"; }
};

class Item_func_vec_index_search_with_distance final : public Item_str_func {
  String m_value;
  String m_number_buf;

 public:
  Item_func_vec_index_search_with_distance(const POS &pos, PT_item_list *arg_list)
      : Item_str_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override {
    return "vec_index_search_with_distance";
  }
};

class Item_func_vec_index_txn_begin final : public Item_int_func {
 public:
  Item_func_vec_index_txn_begin(const POS &pos) : Item_int_func(pos) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_txn_begin"; }
};

class Item_func_vec_index_txn_pending final : public Item_int_func {
 public:
  Item_func_vec_index_txn_pending(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_txn_pending"; }
};

class Item_func_vec_index_stage_upsert final : public Item_int_func {
 public:
  Item_func_vec_index_stage_upsert(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_stage_upsert"; }
};

class Item_func_vec_index_stage_erase final : public Item_int_func {
 public:
  Item_func_vec_index_stage_erase(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_stage_erase"; }
};

class Item_func_vec_index_txn_commit final : public Item_int_func {
 public:
  Item_func_vec_index_txn_commit(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_txn_commit"; }
};

class Item_func_vec_index_txn_rollback final : public Item_int_func {
 public:
  Item_func_vec_index_txn_rollback(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_txn_rollback"; }
};

class Item_func_vec_index_txn_savepoint final : public Item_int_func {
 public:
  Item_func_vec_index_txn_savepoint(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override { return "vec_index_txn_savepoint"; }
};

class Item_func_vec_index_txn_rollback_to final : public Item_int_func {
 public:
  Item_func_vec_index_txn_rollback_to(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_txn_rollback_to";
  }
};

class Item_func_vec_index_txn_release_savepoint final : public Item_int_func {
 public:
  Item_func_vec_index_txn_release_savepoint(const POS &pos,
                                            PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_index_txn_release_savepoint";
  }
};

class Item_func_vec_debug_truth_store_get_hex final : public Item_str_func {
  String m_value;

 public:
  Item_func_vec_debug_truth_store_get_hex(const POS &pos, PT_item_list *arg_list)
      : Item_str_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  String *val_str(String *str) override;
  const char *func_name() const override {
    return "vec_debug_truth_store_get_hex";
  }
};

class Item_func_vec_debug_truth_store_set_hex final : public Item_int_func {
 public:
  Item_func_vec_debug_truth_store_set_hex(const POS &pos, PT_item_list *arg_list)
      : Item_int_func(pos, arg_list) {}

  bool resolve_type(THD *thd) override;
  longlong val_int() override;
  const char *func_name() const override {
    return "vec_debug_truth_store_set_hex";
  }
};

bool format_vector_index_info_json(
    const vector_index_registry::index_info &info, String *out,
    String *number_buf);
bool format_vector_index_list_json(
    const std::vector<std::string> &index_names, String *out);
bool format_vector_search_result_doc_ids(
    const std::vector<vector_index::search_result> &results, String *out,
    String *number_buf);
bool format_vector_search_result_batches(
    const std::vector<std::vector<vector_index::search_result>> &batches,
    String *out, String *number_buf);
bool format_vector_search_results_with_distance(
    const std::vector<vector_index::search_result> &results, String *out,
    String *number_buf);

#endif  // ITEM_VECTORFUNC_INCLUDED
