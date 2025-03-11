
#pragma once

// This is a reference wrapper for an edge pair
template<typename env_type>
struct env_pair {
    env_type L;
    env_type R;
    void      assert_validity() const;
};