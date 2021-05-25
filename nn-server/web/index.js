app_globals = {}

app_globals.createObjectURL = URL.createObjectURL


seamless_read_cells = {
  "text": [
    "plot"
  ],
  "json": [
    "high_ana_mode",
    "high_comp_threshold_factor",
    "high_comp_threshold_redundancy",
    "high_comp_random_nstructures",
    "high_ana_random_discard_upper",
    "high_ana_random_discard_lower",
    "high_ana_random_mode",
    "high_ana_random_bins",
    "low_ana_mode",
    "low_comp_greedy_poolsize",
    "low_ana_greedy_discard",
    "low_ana_greedy_mode",
    "low_ana_greedy_bins",
    "low_comp_nn_k",
    "low_ana_nn_discard",
    "low_ana_nn_mode",
    "low_ana_nn_bins"
  ]
}
seamless_write_cells = {
  "text": [],
  "json": [
    "high_ana_mode",
    "high_comp_threshold_factor",
    "high_comp_threshold_redundancy",
    "high_comp_random_nstructures",
    "high_ana_random_discard_upper",
    "high_ana_random_discard_lower",
    "high_ana_random_mode",
    "high_ana_random_bins",
    "low_ana_mode",
    "low_comp_greedy_poolsize",
    "low_ana_greedy_discard",
    "low_ana_greedy_mode",
    "low_ana_greedy_bins",
    "low_comp_nn_k",
    "low_ana_nn_discard",
    "low_ana_nn_mode",
    "low_ana_nn_bins"
  ]
}
seamless_auto_read_cells = []

ctx = connect_seamless()
ctx.self.onsharelist = function (sharelist) {
  sharelist.forEach(cell => {
    if (ctx[cell].binary) {
      ctx[cell].onchange = function () {
        content_type = ctx[cell].content_type
        if (content_type === null) content_type = ""
        const v = new Blob([this.value], {type: content_type})
        vm[cell].value = v
        vm[cell].checksum = this.checksum
      }
    }
    else if (seamless_read_cells["json"].indexOf(cell) >= 0) {
      ctx[cell].onchange = function () {
        try {
          const v = JSON.parse(this.value)
          vm[cell].value = v
          vm[cell].checksum = this.checksum
        }
        catch (error) {
          console.log(`Cannot parse server value of cell '${cell}' as JSON`)
        }
      }
    }
    else if (seamless_read_cells["text"].indexOf(cell) >= 0) {
      ctx[cell].onchange = function () {
        vm[cell].value = this.value
        vm[cell].checksum = this.checksum
      }
    }

    if (seamless_auto_read_cells.indexOf(cell) >= 0) {
      ctx[cell].auto_read = true
    }
  })
}
webctx = connect_seamless(null, null, share_namespace="status")
webctx.self.onsharelist = function (sharelist) {
  vis_status = webctx["vis_status"]
  if (!(vis_status === undefined)) {
    vis_status.onchange = function() {      
      let jstatus = JSON.parse(vis_status.value)
      cells = {}
      transformers = {}
      jstatus.nodes.forEach(node => {
        if (node.type == "cell") {
          cells[node.name] = node
        }
        else if (node.type == "transformer") {
          transformers[node.name] = node
        }
      })
      jstatus.cells = cells
      jstatus.transformers = transformers
      vm["STATUS"].value = jstatus
      vm["STATUS"].checksum = vis_status.checksum
    }
  }
}  

function seamless_update(cell, value, encoding) {
  if (!ctx) return
  if (!ctx.self.sharelist) return
  if (ctx.self.sharelist.indexOf(cell) < 0) return
  if (ctx[cell].binary) {
    ctx[cell].set(value)
  }
  else if (encoding == "json") {
    ctx[cell].set(JSON.stringify(value))
  }
  else if (encoding == "text") {
    ctx[cell].set(value)
  }
}


const app = new Vue({
  vuetify: new Vuetify(),
  data() {
    return {
      ...{
        "high_ana_mode": {
          "checksum": null,
          "value": ""
        },
        "high_comp_threshold_factor": {
          "checksum": null,
          "value": 0
        },
        "high_comp_threshold_redundancy": {
          "checksum": null,
          "value": 0
        },
        "high_comp_random_nstructures": {
          "checksum": null,
          "value": 0
        },
        "high_ana_random_discard_upper": {
          "checksum": null,
          "value": 0
        },
        "high_ana_random_discard_lower": {
          "checksum": null,
          "value": 0
        },
        "high_ana_random_mode": {
          "checksum": null,
          "value": ""
        },
        "high_ana_random_bins": {
          "checksum": null,
          "value": 0
        },
        "low_ana_mode": {
          "checksum": null,
          "value": ""
        },
        "low_comp_greedy_poolsize": {
          "checksum": null,
          "value": 0
        },
        "low_ana_greedy_discard": {
          "checksum": null,
          "value": 0
        },
        "low_ana_greedy_mode": {
          "checksum": null,
          "value": ""
        },
        "low_ana_greedy_bins": {
          "checksum": null,
          "value": 0
        },
        "low_comp_nn_k": {
          "checksum": null,
          "value": 0
        },
        "low_ana_nn_discard": {
          "checksum": null,
          "value": 0
        },
        "low_ana_nn_mode": {
          "checksum": null,
          "value": ""
        },
        "low_ana_nn_bins": {
          "checksum": null,
          "value": 0
        },
        "plot": {
          "checksum": null,
          "value": null
        }
      }, 
      ...{
        "STATUS": {
          "checksum": null,
          "value": {}
        }
      }
    }
  },
  methods: {
    METHOD_get_app_globals() {
      return app_globals
    },
    METHOD_file_upload(cellname, file) { 
      if (file === undefined) return
      that = this
      file.arrayBuffer().then(function(buf){
        that[cellname].value = new Blob([new Uint8Array(buf)], {type: file.type })
      })  
    }
    
  },
  watch: {
    "high_ana_mode.value": function (value) {
      seamless_update("high_ana_mode", value, "json")
    },
    "high_comp_threshold_factor.value": function (value) {
      seamless_update("high_comp_threshold_factor", value, "json")
    },
    "high_comp_threshold_redundancy.value": function (value) {
      seamless_update("high_comp_threshold_redundancy", value, "json")
    },
    "high_comp_random_nstructures.value": function (value) {
      seamless_update("high_comp_random_nstructures", value, "json")
    },
    "high_ana_random_discard_upper.value": function (value) {
      seamless_update("high_ana_random_discard_upper", value, "json")
    },
    "high_ana_random_discard_lower.value": function (value) {
      seamless_update("high_ana_random_discard_lower", value, "json")
    },
    "high_ana_random_mode.value": function (value) {
      seamless_update("high_ana_random_mode", value, "json")
    },
    "high_ana_random_bins.value": function (value) {
      seamless_update("high_ana_random_bins", value, "json")
    },
    "low_ana_mode.value": function (value) {
      seamless_update("low_ana_mode", value, "json")
    },
    "low_comp_greedy_poolsize.value": function (value) {
      seamless_update("low_comp_greedy_poolsize", value, "json")
    },
    "low_ana_greedy_discard.value": function (value) {
      seamless_update("low_ana_greedy_discard", value, "json")
    },
    "low_ana_greedy_mode.value": function (value) {
      seamless_update("low_ana_greedy_mode", value, "json")
    },
    "low_ana_greedy_bins.value": function (value) {
      seamless_update("low_ana_greedy_bins", value, "json")
    },
    "low_comp_nn_k.value": function (value) {
      seamless_update("low_comp_nn_k", value, "json")
    },
    "low_ana_nn_discard.value": function (value) {
      seamless_update("low_ana_nn_discard", value, "json")
    },
    "low_ana_nn_mode.value": function (value) {
      seamless_update("low_ana_nn_mode", value, "json")
    },
    "low_ana_nn_bins.value": function (value) {
      seamless_update("low_ana_nn_bins", value, "json")
    },
  },
})

vm = app.$mount('#app')
