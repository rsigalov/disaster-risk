with security_table as (
    select
        secid, date, close as under_price
    from optionm.SECPRD
    where secid = _secid_
    and date >= _start_date_
    and date <= _end_date_
), combined_table as (
(   select
        o.secid, o.date, o.exdate, o.strike_price
    from _data_base_ as o
    left join security_table as s
        on o.secid = s.secid and o.date = s.date
        where o.secid = _secid_
    and o.cp_flag = 'C'
    and o.open_interest > 0
    and o.best_bid > 0
    and o.best_offer - o.best_bid > 0
    and o.ss_flag = '0'
    and o.delta is not null
    and o.impl_volatility is not null
    and o.strike_price/1000 > s.under_price
    and (o.best_offer + o.best_bid)/2 < s.under_price
    and o.exdate - o.date <= 365 * 1
    and o.exdate - o.date > 0
    and o.date >= _start_date_
    and o.date <= _end_date_
    order by o.exdate, o.strike_price
) union (
    select
       o.secid, o.date, o.exdate, o.strike_price
    from _data_base_ as o
    left join security_table as s
        on o.secid = s.secid and o.date = s.date
    where o.secid = _secid_
    and o.cp_flag = 'P'
    and o.open_interest > 0
    and o.best_bid > 0
    and o.best_offer - o.best_bid > 0
    and o.ss_flag = '0'
    and o.delta is not null
    and o.impl_volatility is not null
    and o.strike_price/1000 < s.under_price
    and (o.best_offer + o.best_bid)/2 < o.strike_price/1000
    and (o.best_offer + o.best_bid)/2 >= GREATEST(0, o.strike_price/1000 - s.under_price)
    and o.exdate - o.date <= 365 * 1
    and o.exdate - o.date > 0
    and o.date >= _start_date_
    and o.date <= _end_date_
    order by o.exdate, o.strike_price
)
), cnt_table as (
    select 
        secid, date, exdate,
        count(strike_price) as cnt_opts
    from combined_table
    group by secid, date, exdate
    order by secid, date, exdate
)
select 
    secid, date,
    min(exdate - date - 1) as min_maturity,
    max(exdate - date - 1) as max_maturity
from cnt_table
where cnt_opts >= _min_options_
group by secid, date



