with security_table as (
    select
        secid, date, close as under_price
    from optionm.SECPRD
    where date = opt_date
    and secid = _secid_
)
(
    select
	   o.secid, o.date, o.exdate, o.cp_flag, o.strike_price, o.impl_volatility,
       (o.best_offer + o.best_bid)/2 as mid_price,
       s.under_price
	from OPTIONM.OPPRCD2017 as o
    left join security_table as s
        on o.secid = s.secid and o.date = s.date
	where o.secid = _secid_
    and o.cp_flag = 'C'
    and o.date = opt_date
    and o.exdate = exp_date
    and o.open_interest > 0
    and o.best_bid > 0
    and o.best_offer - o.best_bid > 0
    and o.ss_flag = '0'
    and o.delta is not null
    and o.impl_volatility is not null
    and o.strike_price/1000 > s.under_price
    and (o.best_offer + o.best_bid)/2 < s.under_price
	order by o.exdate, o.strike_price
) union (
	select
	   o.secid, o.date, o.exdate, o.cp_flag, o.strike_price, o.impl_volatility,
       (o.best_offer + o.best_bid)/2 as mid_price,
       s.under_price
	from OPTIONM.OPPRCD2017 as o
    left join security_table as s
        on o.secid = s.secid and o.date = s.date
	where o.secid = _secid_
    and o.cp_flag = 'P'
    and o.date = opt_date
    and o.exdate = exp_date
    and o.open_interest > 0
    and o.best_bid > 0
    and o.best_offer - o.best_bid > 0
    and o.ss_flag = '0'
    and o.delta is not null
    and o.impl_volatility is not null
    and o.strike_price/1000 < s.under_price
    and (o.best_offer + o.best_bid)/2 < o.strike_price/1000
    and (o.best_offer + o.best_bid)/2 >= GREATEST(0, o.strike_price/1000 - s.under_price)
	order by o.exdate, o.strike_price
)
