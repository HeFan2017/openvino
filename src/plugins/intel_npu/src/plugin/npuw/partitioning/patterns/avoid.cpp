// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "avoid.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/add.hpp"
#include "openvino/op/power.hpp"
#include "openvino/op/reduce_mean.hpp"
#include "openvino/op/sqrt.hpp"
#include "openvino/op/util/op_types.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace avoid {

namespace opp = ov::pass::pattern;

//------------------------------------------------------------------------------
// Pattern: RMSNorm, from LLaMa-v2-7b model
//
//            Power     Const
//               :        :
//               V        V
//               ReduceMean
//                    :
//                    V
//                   Add
//                    :
//                    V
//                   Sqrt
//                    :
//                    V
//
RMSNorm::RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
    auto power = opp::wrap_type<ov::op::v1::Power>({opp::any_input(), opp::any_input()});
    auto reduce = opp::wrap_type<ov::op::v1::ReduceMean>({power, opp::wrap_type<ov::op::v0::Constant>()});
    auto add = opp::wrap_type<ov::op::v1::Add>({reduce, opp::any_input()});
    auto sqrt = opp::wrap_type<ov::op::v0::Sqrt>({add});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_power = node_to_output.at(power).get_node_shared_ptr();
        auto matched_reduce = node_to_output.at(reduce).get_node_shared_ptr();
        auto matched_add = node_to_output.at(add).get_node_shared_ptr();
        auto matched_sqrt = node_to_output.at(sqrt).get_node_shared_ptr();

        node_to_gptr->at(matched_power)->avoid(avoid_device);
        node_to_gptr->at(matched_reduce)->avoid(avoid_device);
        node_to_gptr->at(matched_add)->avoid(avoid_device);
        node_to_gptr->at(matched_sqrt)->avoid(avoid_device);
        static int count = 0;
        //node_to_gptr->at(matched_power)->NotFusedAbove();
        count++;
        printf("Found %d power\n", count);

        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(sqrt, "TagRMSNormAvoid"), std::move(callback));
}

//RMSNorm::RMSNorm(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& avoid_device) {
//    auto any_input = opp::any_input();
//    auto input_ids = opp::wrap_type<ov::op::v0::Parameter>();
//    auto convert = opp::wrap_type<ov::op::v0::Convert>({input_ids});
//    auto gather = opp::wrap_type<ov::op::v8::Gather>({any_input, convert, opp::any_input()});
//
//    auto node_to_gptr = snapshot->getNodeToGroupMap();
//
//    // Note: Use [=] to make sure the above objects stay alive in the callback
//    auto callback = [=](ov::pass::pattern::Matcher& m) {
//        auto& node_to_output = m.get_pattern_value_map();
//        auto matched_any_input_node = node_to_output.at(any_input).get_node_shared_ptr();
//        auto matched_convert_node = node_to_output.at(convert).get_node_shared_ptr();
//        auto matched_gather_node = node_to_output.at(gather).get_node_shared_ptr();
//
//        auto matched_gather = std::static_pointer_cast<ov::op::v8::Gather>(matched_gather_node);
//
//        auto gather_shape = matched_gather->input(0).get_shape();
//
//        if (gather_shape[0] == 152064 && gather_shape[1] == 3584)
//        {
//            printf("HFDebug: find the head\n");
//            node_to_gptr->at(matched_convert_node)->avoid(avoid_device);
//            node_to_gptr->at(matched_gather_node)->avoid(avoid_device);
//        }
//
//        return false;  // root hasn't changed
//    };
//    register_matcher(std::make_shared<opp::Matcher>(gather, "TagRMSNormAvoid"), std::move(callback));
//}

}  // namespace avoid
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
