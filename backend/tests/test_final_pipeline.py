"""Regression checks for the actual errors corrected during finalisation."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from app.ml.beat_preparation import prepare_beat, base_rr_at, PREPROCESSING_VERSION
from app.ml.feature_system import extend_rr_features
from app.ml.live_hierarchical import BeatStreamAnalyzer


class PreparationTests(unittest.TestCase):
    def test_local_scaling_is_gain_and_offset_invariant(self):
        t = np.arange(512)/128
        x = np.sin(2*np.pi*1.3*t)+.2*np.sin(2*np.pi*17*t)
        np.testing.assert_allclose(prepare_beat(x),prepare_beat(8*x+900),atol=1e-6)

    def test_unusable_input_is_rejected(self):
        for x in (np.zeros(512),np.full(512,np.nan),np.ones(511)):
            with self.assertRaises(ValueError):
                prepare_beat(x)

    def test_rr_requires_observed_neighbors_and_ignores_later_peaks(self):
        peaks=np.asarray([200,310,440,555,670,780])
        np.testing.assert_array_equal(base_rr_at(peaks,2),base_rr_at(np.append(peaks,999999),2))
        self.assertAlmostEqual(float(base_rr_at(peaks,2)[0]),130/128)
        with self.assertRaises(ValueError):
            base_rr_at(peaks,0)
        with self.assertRaises(ValueError):
            base_rr_at(peaks,len(peaks)-1)

    def test_streamed_inputs_match_offline_preparation_across_chunk_boundaries(self):
        peaks=np.arange(220,3800,113)
        t=np.arange(4200)/128
        signal=(np.sin(2*np.pi*1.1*t)+.1*np.sin(2*np.pi*21*t)).astype(np.float32)
        captured=[]
        def classify(window,rr):
            captured.append((window.copy(),rr.copy()))
            return {"class":"N","is_anomaly":False,"anomaly_probability":.1,
                    "threshold":.5,"confidence":.9,"probabilities":{"N":.9},
                    "classification_mode":"MLII-only"}
        engine=SimpleNamespace(improved_system=SimpleNamespace(preprocessing_version=PREPROCESSING_VERSION,rr_feature_count=15),classify_beat=classify)
        analyzer=BeatStreamAnalyzer(engine)
        # Fix peak locations to isolate input preparation from detector accuracy.
        with patch("app.ml.live_hierarchical.detect_rpeaks",side_effect=lambda s,fs: peaks[peaks<len(s)]):
            for start in range(0,len(signal),173):
                analyzer.append(signal[start:start+173])
        eligible=[i for i in range(1,len(peaks)-1) if peaks[i]+312<=len(signal)]
        self.assertEqual(len(captured),len(eligible))
        for (window,rr),i in zip(captured,eligible):
            np.testing.assert_array_equal(window,prepare_beat(signal[peaks[i]-200:peaks[i]+312]))
            expected=extend_rr_features(base_rr_at(peaks,i),np.diff(peaks[max(0,i-20):i+1])/128)
            np.testing.assert_array_equal(rr,expected)


if __name__ == "__main__":
    unittest.main()
