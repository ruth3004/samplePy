# proofreading_base.py

import os
import sys
import argparse
import napari
import numpy as np
from PyQt5.QtWidgets import QApplication, QPushButton, QHBoxLayout, QWidget
from napari.qt.threading import thread_worker
from scripts.sample_db import SampleDB


class ProofreadingBase:
    def __init__(self, step_name):
        self.STEP_NAME = step_name

    def load_sample_data(self, sample_id, db_path):
        # This method should be overridden in each step-specific script
        raise NotImplementedError("This method should be implemented in the step-specific script.")

    def proofread_samples(self, sample_ids, db_path):
        viewer = napari.Viewer()
        proofreading_widget = self.create_proofreading_widget(viewer, sample_ids, db_path)
        viewer.window.add_dock_widget(proofreading_widget, area='bottom')

        @viewer.bind_key('r')
        def mark_for_redo(viewer):
            sample_id = proofreading_widget.sample_ids[proofreading_widget.current_index]
            proofreading_widget.update_database(sample_id, "redo")

        @viewer.bind_key('p')
        def mark_as_proofread(viewer):
            sample_id = proofreading_widget.sample_ids[proofreading_widget.current_index]
            proofreading_widget.update_database(sample_id, "pr")

        @viewer.bind_key('q')
        def close_viewer(viewer):
            viewer.close()

        napari.run()

    def create_proofreading_widget(self, viewer, sample_ids, db_path):
        # This method can be overridden in step-specific scripts if needed
        return ProofreadingWidget(self, viewer, sample_ids, db_path)

    def main(self):
        parser = argparse.ArgumentParser(description=f"Proofread samples for {self.STEP_NAME}")
        parser.add_argument("-l", "--list", required=True, help="Path to text file containing sample IDs")
        parser.add_argument("--db_path",
                            default=r'\\tungsten-nas.fmi.ch\tungsten\scratch\gfriedri\montruth\sample_db.csv',
                            help="Path to the sample database CSV file")
        args = parser.parse_args()

        with open(args.list, 'r') as f:
            sample_ids = f.read().splitlines()

        self.proofread_samples(sample_ids, args.db_path)


class ProofreadingWidget(QWidget):
    def __init__(self, proofreading_base, viewer, sample_ids, db_path):
        super().__init__()
        self.proofreading_base = proofreading_base
        self.viewer = viewer
        self.sample_ids = sample_ids
        self.current_index = 0
        self.db_path = db_path
        self.sample_db = SampleDB()
        self.sample_db.load(db_path)

        layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        layout.addWidget(self.prev_button)
        layout.addWidget(self.next_button)
        self.setLayout(layout)

        self.prev_button.clicked.connect(self.previous_sample)
        self.next_button.clicked.connect(self.next_sample)

        self.load_current_sample()

    def load_current_sample(self):
        sample_id = self.sample_ids[self.current_index]
        data = self.proofreading_base.load_sample_data(sample_id, self.db_path)

        # Clear existing layers and add new data
        self.viewer.layers.clear()
        for layer_name, layer_data in data.items():
            self.viewer.add_image(layer_data, name=layer_name)

        self.viewer.title = f"Proofreading: {sample_id} ({self.current_index + 1}/{len(self.sample_ids)})"

    def previous_sample(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_sample()

    def next_sample(self):
        if self.current_index < len(self.sample_ids) - 1:
            self.current_index += 1
            self.load_current_sample()

    @thread_worker
    def update_database(self, sample_id, status):
        self.sample_db.update_sample_field(sample_id, self.proofreading_base.STEP_NAME, status)
        self.sample_db.save(self.db_path)
        print(f"Updated {sample_id} status to {status}")