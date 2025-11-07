#include "prof.h"
#include "config/settings.h"
#include "tid/tid.h"
#include "tools/common/log.h"
void tools::common::timer::print_timers() {
    tid::level lvl = settings::timer::level;
    tools::log->info("Printing tid tree level {}", enum2sv(lvl));
    if(settings::itebd::on)
        for(const auto &t : tid::get_tree("iTEBD", lvl)) tools::log->info("{}", t.str());

    if(settings::flbit::on)
        for(const auto &t : tid::get_tree("fLBIT", lvl)) tools::log->info("{}", t.str());

    if(settings::idmrg::on)
        for(const auto &t : tid::get_tree("iDMRG", lvl)) tools::log->info("{}", t.str());

    if(settings::fdmrg::on)
        for(const auto &t : tid::get_tree("fDMRG", lvl)) tools::log->info("{}", t.str());

    if(settings::xdmrg::on) {
        for(const auto &t : tid::get_tree("fDMRG", lvl)) tools::log->info("{}", t.str());
        for(const auto &t : tid::get_tree("xDMRG", lvl)) tools::log->info("{}", t.str());
    }
}