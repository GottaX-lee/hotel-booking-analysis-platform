import numpy as np

DEFAULT_PARAMS = {
    'total_bookings': 117427,
    'avg_adr': 103.48,
    'avg_stay_nights': 3.5,
    'current_cancel_rate': 0.3748,
    'target_cancel_rate': 0.30,
    'intervention_cost_per_booking': 5.0,
    'overbook_cost_per_cancel': 50.0,
}

def calculate_revenue(params: dict = None) -> dict:
    if params is None:
        params = DEFAULT_PARAMS

    t = float(params['total_bookings'])
    adr = float(params['avg_adr'])
    nights = float(params['avg_stay_nights'])
    cur_rate = float(params['current_cancel_rate'])
    tgt_rate = float(params['target_cancel_rate'])
    int_cost = float(params.get('intervention_cost_per_booking', 5.0))
    overbook_cost = float(params.get('overbook_cost_per_cancel', 50.0))

    revenue_per_booking = adr * nights
    cur_cancellations = int(t * cur_rate)
    tgt_cancellations = int(t * tgt_rate)
    saved_cancellations = cur_cancellations - tgt_cancellations

    cur_revenue_loss = cur_cancellations * revenue_per_booking
    tgt_revenue_loss = tgt_cancellations * revenue_per_booking
    revenue_saved = cur_revenue_loss - tgt_revenue_loss

    cur_overbook_loss = cur_cancellations * overbook_cost
    tgt_overbook_loss = tgt_cancellations * overbook_cost
    overbook_saved = cur_overbook_loss - tgt_overbook_loss

    total_intervention_cost = t * int_cost

    net_gain = revenue_saved + overbook_saved - total_intervention_cost
    roi = (net_gain / total_intervention_cost * 100) if total_intervention_cost > 0 else 0

    # 场景对比
    cur_actual_revenue = t * revenue_per_booking - cur_revenue_loss
    tgt_actual_revenue = t * revenue_per_booking - tgt_revenue_loss

    return {
        'input': {
            'total_bookings': int(t),
            'avg_adr': round(adr, 2),
            'avg_stay_nights': round(nights, 1),
            'current_cancel_rate': round(cur_rate, 4),
            'target_cancel_rate': round(tgt_rate, 4),
            'intervention_cost_per_booking': round(int_cost, 2),
            'overbook_cost_per_cancel': round(overbook_cost, 2),
        },
        'current': {
            'cancellations': cur_cancellations,
            'revenue_loss': round(cur_revenue_loss, 2),
            'overbook_loss': round(cur_overbook_loss, 2),
            'total_loss': round(cur_revenue_loss + cur_overbook_loss, 2),
            'actual_revenue': round(cur_actual_revenue, 2),
            'actual_revenue_formatted': f'\u00a5{cur_actual_revenue:,.0f}',
        },
        'target': {
            'cancellations': tgt_cancellations,
            'revenue_loss': round(tgt_revenue_loss, 2),
            'overbook_loss': round(tgt_overbook_loss, 2),
            'total_loss': round(tgt_revenue_loss + tgt_overbook_loss, 2),
            'actual_revenue': round(tgt_actual_revenue, 2),
            'actual_revenue_formatted': f'\u00a5{tgt_actual_revenue:,.0f}',
        },
        'savings': {
            'cancellations_reduced': saved_cancellations,
            'revenue_saved': round(revenue_saved, 2),
            'overbook_saved': round(overbook_saved, 2),
            'total_intervention_cost': round(total_intervention_cost, 2),
            'net_gain': round(net_gain, 2),
            'roi': round(roi, 2),
            'roi_formatted': f'{roi:.1f}%',
            'net_gain_formatted': f'\u00a5{net_gain:,.0f}',
        },
    }

def get_default_params():
    return dict(DEFAULT_PARAMS)
