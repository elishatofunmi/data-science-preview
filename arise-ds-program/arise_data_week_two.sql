# Total number of clients who received their first loans in the month of January 2018.
Query

with newTimeJanCustomers as(
select
   clientId
from `propane-highway-202915.arise.Loans`
where loanNumber = 1 and extract(month from Disbdate) =1)

select distinct(clientId), count(clientId) from newTimeJanCustomers janL 
join `propane-highway-202915.arise.BillPayments` as BP on janL.clientId = BP.customerId
group by clientId




# Number of clients that recieved/took out a loan in the following months of 2018.
with newTimeJanCustomers as(
select
   clientId, extract(month from Disbdate) as month
from `propane-highway-202915.arise.Loans`
where loanNumber = 1 and extract(month from Disbdate) > 1)

select distinct(month), count(distinct(clientId)) as number_of_clients from newTimeJanCustomers janL 
join `propane-highway-202915.arise.BillPayments` as BP on janL.clientId = BP.customerId
group by month order by month
   



# Number of clients who have their bill payment in march 2018
with newTimeJanCustomers as(
select
   customerId
from `propane-highway-202915.arise.BillPayments`
where extract(month from billDate) = 3 and extract(year from billDate) = 2018)

select distinct(BP.customerId) from newTimeJanCustomers janL 
join `propane-highway-202915.arise.BillPayments` as BP on janL.customerId = BP.customerId




# Number of clients who had their bill payment from march till december 2018

with newTimeJanCustomers as(
select
   customerId, extract(month from billDate) as month
from `propane-highway-202915.arise.BillPayments`
where extract(month from billDate) > 3 and extract(year from billDate) = 2018)

select distinct(month), count(distinct(BP.customerId)) as Number_of_client from newTimeJanCustomers janL 
join `propane-highway-202915.arise.BillPayments` as BP on janL.customerId = BP.customerId
group by month




