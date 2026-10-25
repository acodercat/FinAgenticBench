"""Runtime and adversarial contract checks for the 2026-09-24 pair."""
import pandas as pd
import pytest

MODULES = [
    'evals.calculation_reconciliation.foreign_holdings_issuance_calendar_stock_bridge',
    'evals.temporal_path.form_d_offering_lineage_participation_cf_overlap',
]


def test_graph_scope_not_issuer_dedup_or_notice_sum():
    from evals.temporal_path.form_d_offering_lineage_participation_cf_overlap import offering_endpoints
    def row(acc, cik, date, parent=None, amendment=True):
        return dict(ACCESSIONNUMBER=acc, CIK=cik, FILING_DATE=date,
                    PREVIOUSACCESSIONNUMBER=parent, ISAMENDMENT=amendment,
                    SUBMISSIONTYPE='D/A' if amendment else 'D')
    rows = [row('r1','0001','2024-01-01',amendment=False),
            row('r2','0001','2024-02-01',amendment=False),
            row('a1','0001','2024-03-01','r1'), row('a2','0001','2024-03-01','r1'),
            row('late','0001','2025-01-01','a2'), row('cross','0002','2024-09-01','r1'),
            row('orphan','0001','2024-04-01','absent'), row('cycle1','0001','2024-05-01','cycle2'),
            row('cycle2','0001','2024-05-01','cycle1'),
            row('old','0003','2023-01-01',amendment=False),row('old_a','0003','2024-03-01','old')]
    frame = pd.DataFrame(rows)
    assert set(offering_endpoints(frame).ACCESSIONNUMBER) == {'a2','r2'}
    assert set(offering_endpoints(frame.sample(frac=1,random_state=42)).ACCESSIONNUMBER) == {'a2','r2'}
    with pytest.raises(ValueError, match='Duplicate'):
        offering_endpoints(pd.concat([frame, frame.iloc[:1]]))


